import numpy as np

import tensorflow as tf
import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))


b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder("float", [None,10])

with tf.device('/cpu:0'):
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%100==0:
        print i,sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
