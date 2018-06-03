# coding:utf-8
import tensorflow as tf
import numpy as np
import time

import examples.utils as utils

PATH = r"D:\Codes\git\stanford-tensorflow-tutorials\examples\data\MNIST"
N_TRAIN = 60000
N_TEST = 10000

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCH = 30


train, valid, test = utils.read_mnist(PATH)

train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.batch(BATCH_SIZE)
test_data= tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(BATCH_SIZE)
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# only one layer
weights = tf.get_variable("weights", shape=(784,10),initializer=tf.random_normal_initializer(0,0.1))
bias = tf.get_variable("bias", shape=(1, 10),initializer=tf.zeros_initializer())
logits = tf.sigmoid(tf.matmul(img, weights) + bias)

# # two layers
# weights1 = tf.get_variable("weights1", shape=(784, 100), initializer=tf.random_normal_initializer(0,0.1))
# bias1 = tf.get_variable("bias1", shape=(1,100), initializer=tf.zeros_initializer())
# output1 = tf.sigmoid(tf.matmul(img, weights1) + bias1)
#
# weights2 = tf.get_variable("weights2", shape=(100,10), initializer=tf.random_normal_initializer(0,0.1))
# bias2 = tf.get_variable("bias2", shape=(1,10), initializer=tf.zeros_initializer())
# logits = tf.matmul(output1, weights2) + bias2

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name="entropy")
loss = tf.reduce_mean(entropy, name="loss")

# # use GradientDescentOptimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
# use AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)


preds = tf.nn.softmax(logits)
correct = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))

writer = tf.summary.FileWriter(logdir="./graphs/logistic_reg", graph=tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print("Average loss in epoch {}: {}".format(i, total_loss/n_batches))
    print("Train time is {} secondes.".format(time.time()-start_time))

    sess.run(test_init)
    total_correct = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass
    print("Accuracy is {:.3%}.".format(total_correct/N_TEST))
writer.close()



