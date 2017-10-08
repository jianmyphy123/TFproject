import os

import tensorflow as tf


sess = tf.Session()

hello = tf.constant('Hello, Tensorflow')
print(sess.run(hello))
# b'Hello, Tensorflow'

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
node4 = node1 + node2
print(sess.run(node3))
print(sess.run(node4))
# 7.0 7.0

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = tf.add(a, b)
print(sess.run(adder_node, feed_dict={a: 3, b:4.5}))
print(sess.run(adder_node, feed_dict={a: [1,2], b: [3,4]}))
# 7.5
# [ 4.  6.]
