import tensorflow as tf
import data

#Settings
batchSize = 500
useItemsToPredict = 100
futureItemsToPredictMean = 3

# Create the model
x = tf.placeholder(tf.float32, [None, 100])
W = tf.Variable(tf.zeros([100, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(100):
    batch_xs, batch_ys = data.getTrainingBatch(batchSize, useItemsToPredict, futureItemsToPredictMean)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_xs, batch_ys = data.getTrainingBatch(batchSize, useItemsToPredict, futureItemsToPredictMean)
print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))