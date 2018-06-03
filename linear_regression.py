# coding:utf-8
import tensorflow as tf
import numpy as np
import time

def huber_loss(labels, predictions, delta=tf.constant([14.0])):
    residual = tf.abs(labels-predictions)
    loss1 = 0.5 * tf.square(residual)
    loss2 = delta * residual - 0.5 * tf.square(delta)
    return tf.where(residual < delta, loss1, loss2)

PATH = "../examples/data/birth_life_2010.txt"

data = np.loadtxt(PATH, delimiter="	", dtype="str", skiprows=1)
input_x = data[:, 1].astype("float32").tolist()
input_y = data[:, 2].astype("float32").tolist()
n_samples = data.shape[0]

start_time = time.time()

# use tf.placeholder
X = tf.placeholder(tf.float32, shape=[1], name="X")
Y = tf.placeholder(tf.float32, shape=[1], name="Y")

# # use tf.data.Dataset
# dataset = tf.data.Dataset.from_tensor_slices((input_x, input_y))
# iterator = dataset.make_initializable_iterator()
# X, Y = iterator.get_next()

weight = tf.get_variable("weight", dtype=tf.float32, initializer=tf.random_normal(shape=[1]))
bias = tf.get_variable("bias", dtype=tf.float32, initializer=tf.random_normal(shape=[1]))
y_pre = X * weight + bias

# use mean square error
loss = tf.square(Y - y_pre, name="loss")

# # use hubor loss
# loss = huber_loss(Y, y_pre)

# use GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# # use AdamOptimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graph/linear_reg", sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        total_loss = 0
        # use tf.placeholder
        for x, y in zip(input_x, input_y):
            _, l = sess.run([optimizer, loss], feed_dict={X:[x], Y:[y]})
            total_loss += l
        print("Epoch {} : {}".format(i, total_loss/n_samples))

        # # use tf.data.Dataset
        # sess.run(iterator.initializer)
        # try:
        #     while True:
        #         _, l = sess.run([optimizer, loss])
        #         total_loss += l
        # except tf.errors.OutOfRangeError:
        #     pass
        # print("Epoch {} : {}".format(i, total_loss / n_samples))

    writer.close()
    w_out, b_out = sess.run([weight, bias])

end_time = time.time()
print("Cost {} secondes.".format(end_time-start_time))

plt.figure()
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()


