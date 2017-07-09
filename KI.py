import tensorflow as tf
from tensorflow.contrib import rnn
import data

batch_xs, batch_ys = data.getTrainingBatch(1)


n_input = batch_xs[0].__len__()
n_output = batch_ys[0].__len__()
epochs = 500
batchSize = 100
displaySteps = 10
learnRate = 0.01


# Create the model
x = tf.placeholder(tf.float32, [None, n_input])
W = tf.Variable(tf.zeros([n_input, n_output]))
b = tf.Variable(tf.zeros([n_output]))
y = tf.matmul(x, W) + b
pred = tf.nn.softmax(y)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_output])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
acc = 0
for i in range(epochs):
    batch_xs, batch_ys = data.getTrainingBatch(batchSize)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    acc += sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

    if i%displaySteps==0:
        print("Accuracy: ", acc/(i+1), round(i/epochs*100), "% done")

print("Training done")

def predictBuy(data):
    result = sess.run(pred, feed_dict={x:data})
    if result[0][0] > result[0][1]: return True
    else: return False

lastBuy = False
win = 0
lastQuant = 0
for pos in range(data.len):
    prices = [data.prices[pos:pos+n_input]]
    priceNow = prices[0][-1]

    # long
    if lastBuy:
        if priceNow>lastBuy*1.01 or priceNow<lastBuy*0.99:
            winChange = (priceNow-lastBuy)*lastQuant
            win += winChange
            print("made ",winChange,"at",priceNow,"now",win)
            lastBuy = False

    # short
    elif predictBuy(prices):
        #print("Buy  at",priceNow)
        lastBuy = priceNow
        lastQuant = 1/priceNow

print()