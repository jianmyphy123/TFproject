# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(X, W) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = { X: batch_xs, Y: batch_ys }
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.09f}'.format(avg_cost))

print('Learning Finished')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={ X: mnist.test.images, Y: mnist.test.labels }))


# Extracting MNIST_data/train-images-idx3-ubyte.gz
# Extracting MNIST_data/train-labels-idx1-ubyte.gz
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
# Epoch: 0001 cost = 5.745170991
# Epoch: 0002 cost = 1.780056699
# Epoch: 0003 cost = 1.122778631
# Epoch: 0004 cost = 0.872012249
# Epoch: 0005 cost = 0.738203189
# Epoch: 0006 cost = 0.654728886
# Epoch: 0007 cost = 0.596023609
# Epoch: 0008 cost = 0.552216816
# Epoch: 0009 cost = 0.518254964
# Epoch: 0010 cost = 0.491113193
# Epoch: 0011 cost = 0.468347532
# Epoch: 0012 cost = 0.449374347
# Epoch: 0013 cost = 0.432675656
# Epoch: 0014 cost = 0.418828155
# Epoch: 0015 cost = 0.406128934
# Learning Finished
# Accuracy:  0.9023
