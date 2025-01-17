#test this code on GPU server
import input_data
import tensorflow as tf
import numpy as np
# input data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# variables
def weight_variable(shape,name = "W"):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name = name)
def bias_variable(shape,name = "b"):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name = name)
# network operations
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# initialize placeholders
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
keep_prob = tf.placeholder("float") #for dropout

# initialize variables
W_conv1 = weight_variable([5, 5, 1, 32],name = "W_conv1")
b_conv1 = bias_variable([32],name = "b_conv1")
W_conv2 = weight_variable([5, 5, 32, 64],name = "W_conv2")
b_conv2 = bias_variable([64],name = "b_conv2")
W_fc1 = weight_variable([7 * 7 * 64, 1024],name = "W_fc1")
b_fc1 = bias_variable([1024],name = "b_fc1")
W_fc2 = weight_variable([1024, 10],name = "W_fc2")
b_fc2 = bias_variable([10],name = "b_fc2")

with tf.device('/cpu:0'):
    # network operations
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#output of the network
    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))#the objective function
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#the optimizer
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#for testing

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#REVIEW:reduce_mean should be outside of the GPU?

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "cnn_model1.ckpt")
    # Testing :
    print sess.run(accuracy,feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
