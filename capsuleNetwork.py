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