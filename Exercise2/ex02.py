"""A deep MNIST classifier using convolutional layers.
Based on the TensorFlow tutorial: https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

FLAGS = None

# parameters
FILTER_DEPTH = 16
FILTER_SIZE = 3
FCL_UNITS = 128
NUMBER_OF_CLASSES = 10
LEARNING_RATE = 0.1
EPOCHS = 20
BATCH_SIZE = 50

def deepnn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer 
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([FILTER_SIZE, FILTER_SIZE, 1, FILTER_DEPTH])
    b_conv1 = bias_variable([FILTER_DEPTH])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_DEPTH, FILTER_DEPTH])
    b_conv2 = bias_variable([FILTER_DEPTH])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * FILTER_DEPTH, FCL_UNITS])
    b_fc1 = bias_variable([FCL_UNITS])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*FILTER_DEPTH])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([FCL_UNITS, NUMBER_OF_CLASSES])
    b_fc2 = bias_variable([NUMBER_OF_CLASSES])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('sgd'):
    sgd_optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  validation_accuracy = []
  output_file = open('out.csv', 'w')
    
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for epoch in range(EPOCHS):
      for i in range (int(len(mnist.train.labels) / BATCH_SIZE)):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        if i % 100 == 0:
            print("Epoch:", (epoch + 1), " {:.1f} % of batch".format((i/int(len(mnist.train.labels) / BATCH_SIZE))*100) , " training accuracy: {:.1f}%".format(train_accuracy*100))
        sgd_optimiser.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
      
      
      validation_accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
      print("Epoch:", (epoch + 1), " Validation accuracy: {:.3f}".format(validation_accuracy))
      output_file.write("{:};".format(epoch+1))
      output_file.write("{:.3f};\n".format(validation_accuracy))

    output_file.close()

    print("Test accuracy", (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
    print('Duration: {:.1f}s'.format(time.time()-start_time))
    
    #compute number of parameters
    number_of_parameters = np.sum([np.prod(variable.get_shape().as_list()) for variable in tf.trainable_variables()])
    print("Number of parameters: ", number_of_parameters)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
