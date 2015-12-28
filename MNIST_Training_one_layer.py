import numpy as np

import tensorflow as tf
import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



with tf.device('/cpu:0'):
    x = tf.placeholder("float", [None, 784])

    W = tf.Variable(tf.zeros([784,10]))

    W_visualize = tf.reshape(tf.transpose(W,[1,0]), [10,28,28,1])

    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b)

    y_ = tf.placeholder("float", [None,10])

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

np.shape(mnist.train.images)

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%10==0:
        print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
