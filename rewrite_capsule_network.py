import numpy as np
import tensorflow as tf
import tempfile
from tensorflow.examples.tutorials.mnist import input_data

data_dir = '/tmp/tensorflow/mnist/input_data'
num_iterations = 3
batch_size = 128 
lambda_val = 0.5
m_plus = 0.9
m_minus = 0.1
lr = 1e-4


#def capsule(input_data, b_ij, ind_j):
def capsule(input_data, b_ij):
	"""builds the capsules for use in the capsule network as described by Hinton et all
	Args:
		input_data: input tensor of image data
		b_ij: TODO
		ind_j: current index
	Returns:
		Capsules for routing v_j and original index b_ij
	"""
	"""w = tf.Variable(np.random.normal(size = [1, 1152, 10, 8, 16], scale = 0.01), dtype = tf.float32)
	w_ij = tf.tile(w, [batch_size, 1, 1, 1, 1]) # w batch_size times: [batch_size, 1152, 8, 16]
	tiled_data = tf.tile(input_data, [1, 1, 10, 1, 1])

	u_hat = tf.matmul(w_ij, input_data, transpose_a = True) # [batch_size, 1152, 16, 1]
	#shape = [ind_j, 1, b_ij.get_shape().as_list()[2] - ind_j - 1]
	u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')"""

	print('input shape', input_data.get_shape())
	W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
	                    initializer=tf.random_normal_initializer(stddev=0.01))

	# input => [batch_size, 1152, 10, 8, 1]
	# W => [batch_size, 1152, 10, 8, 16]
	input_data = tf.tile(input_data, [1, 1, 10, 1, 1])
	W = tf.tile(W, [batch_size, 1, 1, 1, 1])
	assert input_data.get_shape() == [batch_size, 1152, 10, 8, 1]

	u_hat = tf.matmul(W, input_data, transpose_a=True)
	assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]

	u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
	for r in range(0, num_iterations):
	    # line 4:
	    c_ij = tf.nn.softmax(b_ij, dim = 2) # probability distribution of shape [1, 1152, 10, 1]
	    
	    if r == num_iterations - 1:
	    	# line 5 and 6
	    	v_j = squash(tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=1, keep_dims=True))
	    
	    elif r < num_iterations - 1:
		    # line 5:
		    #b_il, b_ij, b_ir = tf.split(b_ij, shape, axis = 2)
		    #c_il, c_ij, c_ir = tf.split(c_ij, shape, axis = 2) # [1, 1152, 1, 1]

		    # line 6
		    v = squash(tf.reduce_sum(tf.multiply(c_ij, u_hat_stopped), axis = 1, keep_dims = True)) # squash with Eq.1: [batch_size, 1, 16, 1]

		    # line 7
		    v_j = tf.tile(v, [1, 1152, 1, 1, 1]) # now [batch_size, 1152, 16, 1]
		    u_v = tf.matmul(u_hat_stopped, v_j, transpose_a = True) # [batch_size, 1152, 1, 1]

		    #b_ij += tf.reduce_sum(u_v, axis = 0, keep_dims = True)
		    #b_ij = tf.concat([b_il, b_ij, b_ir], axis = 2)
		    b_ij += u_v

	return v


def squash(input_vec):
    """vec = tf.sqrt(tf.reduce_sum(tf.square(input_vec)))  # scalar
    temp = tf.square(vec) / (1 + tf.square(vec))
    squashed = temp * tf.divide(input_vec, vec)  # elementwise
    return squashed"""
    epsilon = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(input_vec), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * input_vec  # element-wise
    return(vec_squashed)
   


def build_capsules(routing, input_data, output_vec_len, num_capsules, kernel_size = None, stride = None):
	""" Assembles the capsules with or without routing depending on layer.
	Args:
		routing: boolean determining whether this is the primary or digit layer
		input_data: input tensor of image data
		output_vec_len: desired output length for this capsule set
		num_capsules: number of output capsules
		kernel_size: for CNN-like behavior
		stride: for CNN-like behavior
	Returns:
		The list of assembled capsules
	"""
	if routing: # digit layer
		"""input_data = tf.reshape(input_data, shape = (batch_size, 1152, 8, 1))
		b_ij = tf.zeros(shape = [1, 1152, 10, 1], dtype = np.float32)
		all_capsules = []
		for j in range(num_capsules):
		    with tf.variable_scope('caps_' + str(j)):
		        caps_j, b_ij = capsule(input_data, b_ij, j)
		        all_capsules.append(caps_j)

		all_capsules = tf.concat(all_capsules, axis = 1) # [batch_size, 10, 16, 1]"""
		input_data = tf.reshape(input_data, shape=(batch_size, -1, 1, input_data.shape[-2].value, 1))

		b_IJ = tf.constant(np.zeros([batch_size, input_data.shape[1].value, num_capsules, 1, 1], dtype=np.float32))
		capsules = capsule(input_data, b_IJ)
		capsules = tf.squeeze(capsules, axis=1)

	else: # primary layer
		"""capsules = []
		for i in range(0, output_vec_len):
		    with tf.variable_scope('unit_' + str(i)):
		        curr_capsule = tf.contrib.layers.conv2d(input_data, num_capsules, kernel_size, stride, padding = "VALID")
		        curr_capsule = tf.reshape(curr_capsule, shape = (batch_size, -1, 1, 1))
		        capsules.append(curr_capsule) # each capsule is [batch_size, 6, 6, 32]

		# [batch_size, 1152, 8, 1]
		all_capsules = tf.concat(capsules, axis = 2)
		all_capsules = squash(all_capsules)"""
		capsules = tf.contrib.layers.conv2d(input_data, num_capsules * output_vec_len,
		                                    kernel_size, stride, padding="VALID",
		                                    activation_fn=tf.nn.relu)
		capsules = tf.reshape(capsules, (batch_size, -1, output_vec_len, 1))

		# [batch_size, 1152, 8, 1]
		capsules = squash(capsules)
		assert capsules.get_shape() == [batch_size, 1152, 8, 1]

	return capsules


def loss(v_k, batch_size, X, Y, pred):
	# margin loss
    max1 = tf.reshape(tf.square(tf.maximum(0., m_plus - v_k)), shape = (batch_size, -1))
    max2 = tf.reshape(tf.square(tf.maximum(0., v_k - m_minus)), shape = (batch_size, -1))

    L_k = Y * max1 + lambda_val * (1 - Y) * max2
    margin_loss = tf.reduce_mean(tf.reduce_sum(L_k, axis = 1))

    # reconstruction loss
    correct = tf.reshape(X, shape = (batch_size, -1))
    squared = tf.square(pred - correct)
    reconstruction_loss = tf.reduce_mean(squared)

    # total loss
    total_loss = margin_loss + 0.0005 * reconstruction_loss

    return total_loss


def capsule_network(X, Y):
	"""Builds the actual capsule network
	Args:
		X: an input tensor with image data
		Y: an input tensor with label data
	Returns:
		Predicted label and loss from this iteration
	"""
	epsilon = 1e-9
	convolution = tf.contrib.layers.conv2d(X, num_outputs = 256, kernel_size = 9, stride = 1, padding = 'VALID')
	assert convolution.get_shape() == [batch_size, 20, 20, 256]
	capsule_one = build_capsules(False, convolution, 8, 32, kernel_size = 9, stride = 2)
	capsule_two = build_capsules(True, capsule_one, 16, 10)

	v_k = tf.sqrt(tf.reduce_sum(tf.square(capsule_two), axis = 2, keep_dims = True) + epsilon) # calculate ||v_k||
	softmax_v = tf.nn.softmax(v_k, dim = 1)

	largest_ind = tf.reshape(tf.argmax(softmax_v, axis = 1, output_type = tf.int32), shape = (batch_size, ))

	masked = []
	for batch in range(0, batch_size): # TODO: example code looks buggy here with "for batchsize in range(batchsize)"...
	    v = capsule_two[batch][largest_ind[batch], :]
	    masked.append(tf.reshape(v, shape = (1, 1, 16, 1)))

	masked = tf.concat(masked, axis = 0)

	j_vec = tf.reshape(masked, shape = (batch_size, -1))
	output_1 = tf.contrib.layers.fully_connected(j_vec, num_outputs = 512)
	output_2 = tf.contrib.layers.fully_connected(output_1, num_outputs = 1024)
	pred = tf.contrib.layers.fully_connected(output_2, num_outputs = 784, activation_fn = tf.sigmoid)

	total_loss = loss(v_k, batch_size, X, Y, pred)

	return pred, total_loss


def run_model():
  mnist = input_data.read_data_sets(data_dir, one_hot=True)

  x = tf.placeholder(tf.float32, shape = (batch_size, 28, 28, 1))
  y = tf.placeholder(tf.float32, shape = (batch_size, 10))

  pred, loss = capsule_network(x, y)

  global_step = tf.Variable(0, name = 'global_step', trainable = False)
  train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step = global_step)

  correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(batch_size)
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict = {x: np.reshape(batch[0], [-1, 28, 28, 1]), y: batch[1]})
			print('step %d, training accuracy %g' % (i, train_accuracy))
		train_step.run(feed_dict = {x: np.reshape(batch[0], [-1, 28, 28, 1]), y: batch[1]})

	print('test accuracy %g' % accuracy.eval(feed_dict = {x: np.reshape(mnist.test.images, [-1, 28, 28, 1]), y: mnist.test.labels}))


run_model()