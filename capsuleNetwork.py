import numpy as np
import tensorflow as tf

output_vec_len = 10 # TODO
num_iterations = 3
batch_size = 128 
lambda_val = 0.5
m_plus = 0.9
m_minus = 0.1
lr = 1e-4

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


def build_capsules(routing, input_data, num_units, num_capsules, kernel_size=None, stride=None):
	if routing: # digit layer
		# reshape to [batch_size, 1152, 8, 1]
	    data = tf.reshape(input_data, shape = (batch_size, 1152, 8, 1)) # TODO: why is this reshaped variable never used in the example code?

	    # b_ij: [1, num_caps_l, num_caps_l_plus_1, 1]
	    b_ij = tf.zeros(shape = [1, 1152, 10, 1], dtype = np.float32)
	    capsules = []
	    for j in range(num_capsules):
	        with tf.variable_scope('caps_' + str(j)):
	            curr, b_ij = capsule(input_data, b_ij, j)
	            capsules.append(curr)

	    all_capsules = tf.concat(capsules, axis = 1) # [batch_size, 10, 16, 1]

	else: # primary layer
	    capsules = []
	    for i in range(0, output_vec_len):
	        with tf.variable_scope('unit_' + str(i)):
	            curr_capsule = tf.contrib.layers.conv2d(input_data, num_capsules, kernel_size, stride, padding="VALID")
	            curr_capsule = tf.reshape(curr_capsule, shape = (batch_size, -1, 1, 1))
	            capsules.append(curr_capsule) # each capsule is [batch_size, 6, 6, 32]

	    # [batch_size, 1152, 8, 1]
	    all_capsules = tf.concat(capsules, axis = 2)
	    all_capsules = squash(all_capsules)	    

	return(all_capsules)


def capsule_network():
	with tf.variable_scope('convolution_1'): # [batch_size, 20, 20, 256]
	    conv1 = tf.contrib.layers.conv2d(X, num_outputs = 256, kernel_size = 9, stride = 1, padding='VALID')

	with tf.variable_scope('primary_capsule'): # [batch_size, 1152, 8, 1]
	    caps1 = build_capsules(routing=False, conv1, num_units = 8, num_capsules = 32, kernel_size=9, stride=2)

	with tf.variable_scope('digit_capsule'): # [batch_size, 10, 16, 1]
	    caps2 = build_capsules(routing=True, caps1, num_units=16, num_capsules=10)

	with tf.variable_scope('masking'):
		# TODO: make v_c accessible below
	    v_c = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True)) # calculate ||v_c||
	    softmax_v = tf.nn.softmax(v_c, dim=1)

	    largest_ind = tf.argmax(softmax_v, axis=1, output_type=tf.int32)

	    masked_v = []
	    largest_ind = tf.reshape(largest_ind, shape=(batch_size, ))
	    
	    for batch in range(0, batch_size): # TODO: example code looks buggy here with "for batchsize in range(batchsize)"...
	        v = caps2[batch][largest_ind[batch], :]
	        masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

	    masked_v = tf.concat(masked_v, axis=0)

	with tf.variable_scope('fully_connected'): # fully connected layers
	    j_vec = tf.reshape(masked_v, shape=(batch_size, -1))
	    fc1 = tf.contrib.layers.fully_connected(j_vec, num_outputs=512)
	    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
	    pred = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)


def run_model():
  x = tf.placeholder(tf.float32, shape = (batch_size, 28, 28, 1))
  y_ = tf.placeholder(tf.float32, shape = (batch_size, 28, 28, 1))
  y_pred, keep_prob = capsule_network(x)

  with tf.name_scope('loss'):
    # margin loss
    max_l = tf.square(tf.maximum(0., m_plus - v_c)) # max(0, m_plus - ||v_c||)^2
    max_r = tf.square(tf.maximum(0., v_c - m_minus)) # max(0, ||v_c|| - m_minus)^2

    max_l = tf.reshape(max_l, shape = (batch_size, -1))
    max_r = tf.reshape(max_r, shape = (batch_size, -1))

    T_c = Y # TODO: check this
    L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r
    margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

    # reconstruction loss
    correct = tf.reshape(X, shape=(batch_size, -1))
    squared = tf.square(pred - correct)
    reconstruction_loss = tf.reduce_mean(squared)

    # total loss
    total_loss = margin_loss + 0.0005 * reconstruction_loss

    # summarize
    tf.summary.scalar('margin_loss', margin_loss)
    tf.summary.scalar('reconstruction_loss', reconstruction_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.image('reconstruction_img', tf.reshape(pred, shape=(batch_size, 28, 28, 1)))
    merged_sum = tf.summary.merge_all()

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
			print('step %d, training accuracy %g' % (i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


run_model()