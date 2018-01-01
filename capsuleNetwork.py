import numpy as np
import tensorflow as tf

num_capsules = 4 # TODO
output_vec_len = 10 # TODO
kernel_size = 10 # TODO
stride = 1 # TODO
num_iterations = 4 # TODO
batch_size = 1000 # TODO

def capsule(input_data, b_ij, ind_j):
	with tf.variable_scope('routing'): # TODO: example code has this variable_scope -- do I actually need it?
	    w_ij = tf.Variable(np.random.normal(size=[1, 1152, 8, 16], scale=0.01), dtype = tf.float32)
	    w_ij = tf.tile(w_ij, [batch_size, 1, 1, 1]) # w_ij batch_size times: [batch_size, 1152, 8, 16]

	    u_hat = tf.matmul(w_ij, input_data, transpose_a=True) # [8, 16].T x [8, 1]: [16, 1]

	    shape = b_ij.get_shape().as_list()

	    split = [ind_j, 1, shape[2] - ind_j - 1] # TODO: what is this used for?

	    for r in range(0, num_iterations):
	        # line 4:
	        c_ij = tf.nn.softmax(b_if, dim=2) # probability distribution of shape [1, 1152, 10, 1]

	        # line 5:
	        b_il, b_ij, b_ir = tf.split(b_ij, split, axis=2)
	        c_il, c_ij, b_or = tf.split(c_ij, split, axis=2)

	        # line 6
	        v_j = squash(tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=1, keep_dims=True)) # squash using Eq.1, resulting in [batch_size, 1, 16, 1]

	        # line 7
	        v_j = tf.tile(v_j, [1, 1152, 1, 1]) # now [batch_size, 1152, 16, 1]
	        u_v = tf.matmul(u_hat, v_j, transpose_a=True)
	        b_ij += tf.reduce_sum(u_v, axis=0, keep_dims=True) # reduce in the batch_size dim: [1, 1152, 1, 1]
	        b_ij = tf.concat([b_il, b_ij, b_ir], axis=2)

	    return(v_j, b_ij)


def squash(input_vec):
    vec = tf.sqrt(tf.reduce_sum(tf.square(input_vec)))  # scalar
    vec_updated = tf.square(vec) / (1 + tf.square(vec))
    squashed = vec_updated * tf.divide(input_vec, vec)  # elementwise
    return(squashed)


def build(routing, batch_size, input_data):
	if routing: # digit layer
		# reshape to [batch_size, 1152, 8, 1]
	    data = tf.reshape(input_data, shape=(batch_size, 1152, 8, 1)) # TODO: why is this reshaped variable never used in the example code?

	    # b_ij: [1, num_caps_l, num_caps_l_plus_1, 1]
	    b_ij = tf.zeros(shape=[1, 1152, 10, 1], dtype=np.float32)
	    capsules = []
	    for j in range(num_capsules):
	        with tf.variable_scope('caps_' + str(j)):
	            curr, b_ij = capsule(input_data, b_ij, j)
	            capsules.append(curr)

	    # form tensor of shape [batch_size, 10, 16, 1]
	    all_capsules = tf.concat(capsules, axis=1)

	else: # primary layer
	    # [batch_size, 20, 20, 256]
	    capsules = []
	    for i in range(0, output_vec_len):
	        # each capsule i: [batch_size, 6, 6, 32]
	        with tf.variable_scope('unit_' + str(i)):
	            curr_capsule = tf.contrib.layers.conv2d(input_data, num_capsules, kernel_size, stride, padding="VALID")
	            curr_capsule = tf.reshape(curr_capsule, shape=(batch_size, -1, 1, 1))
	            capsules.append(curr_capsule)

	    # [batch_size, 1152, 8, 1]
	    all_capsules = tf.concat(capsules, axis=2)
	    all_capsules = squash(all_capsules)	    

	return(all_capsules)


def loss(X, Y, pred):
	# margin loss
    max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length)) # max(0, m_plus-||v_c||)^2
    max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus)) # max(0, ||v_c||-m_minus)^2

    max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
    max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

    T_c = Y # TODO: check this
    L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
    margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

    # reconstruction loss
    correct = tf.reshape(X, shape=(batch_size, -1))
    squared = tf.square(pred - correct)
    reconstruction_loss = tf.reduce_mean(squared)

    # total loss
    total_loss = margin_loss + 0.0005 * reconstruction_err

    # summarize
    tf.summary.scalar('margin_loss', margin_loss)
    tf.summary.scalar('reconstruction_loss', reconstruction_err)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.image('reconstruction_img', tf.reshape(pred, shape=(batch_size, 28, 28, 1)))
    merged_sum = tf.summary.merge_all()

    return total_loss


def network():
	with tf.variable_scope('convolution_1'): # Conv1, [batch_size, 20, 20, 256]
	    conv1 = tf.contrib.layers.conv2d(X, num_outputs=256, kernel_size=9, stride=1, padding='VALID')

	with tf.variable_scope('primary_capsule'): # Primary Capsules, [batch_size, 1152, 8, 1]
	    primaryCaps = CapsConv(num_units=8, with_routing=False)
	    caps1 = primaryCaps(conv1, num_outputs=32, kernel_size=9, stride=2)

	with tf.variable_scope('digit_capsule'): # DigitCaps layer, [batch_size, 10, 16, 1]
	    digitCaps = CapsConv(num_units=16, with_routing=True)
	    caps2 = digitCaps(caps1, num_outputs=10)

	with tf.variable_scope('masking'):
	    v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True)) # calculate ||v_c||
	    softmax_v = tf.nn.softmax(v_length, dim=1)

	    argmax_ind = tf.argmax(softmax_v, axis=1, output_type=tf.int32)

	    masked_v = []
	    argmax_ind = tf.reshape(argmax_ind, shape=(batch_size, ))
	    for batch in range(0, batch_size): # TODO: example code looks buggy here with "for batchsize in range(batchsize)"...
	        v = caps2[batch][argmax_ind[batch], :]
	        masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

	    masked_v = tf.concat(masked_v, axis=0)

	with tf.variable_scope('fully_connected'): # fully connected layers
	    j_vec = tf.reshape(masked_v, shape=(batch_size, -1))
	    fc1 = tf.contrib.layers.fully_connected(j_vec, num_outputs=512)
	    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
	    pred = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)





