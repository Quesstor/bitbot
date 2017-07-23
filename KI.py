import tensorflow as tf
from tensorflow.contrib import rnn
import data
import numpy as np
batch_xs, batch_ys = data.getTrainingBatch(1)


n_input = batch_xs[0].__len__()
n_output = batch_ys[0].__len__()

n_hidden_1 = n_input
n_hidden_2 = n_input

epochs = 100
batchSize = 200
displaySteps = 1
learnRate = 0.01



# Create model
def multilayer_perceptron(input, n_input, n_hidden_1, n_hidden_2, n_output):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



input = tf.placeholder(tf.float32, [None, n_input])
pred = multilayer_perceptron(input, weights, biases)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_output])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred))
train_step = tf.train.AdamOptimizer(learnRate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
acc = 0
try:
    for i in range(epochs):
        batch_xs, batch_ys = data.getTrainingBatch(batchSize)
        _, d, p = sess.run([train_step, accuracy, pred], feed_dict={input: batch_xs, y_: batch_ys})
        acc += d

        if i%displaySteps==0:
            print("Accuarcy: ", acc/(i+1), round(i/epochs*100), "% done. Outputcounts: ", data.ouputcounts)
except Exception as e:
    print("Training stoped: "+str(e))

print("Training done")

def predictBuy(data):
    result = sess.run(pred, feed_dict={input:[data]})
    test = np.argmax(result)
    return test>12


def runSimulation():
    global data
    print("Simulating")
    winChanges = list()
    lastBuy = False
    win = 0
    lastQuant = 0
    for i in range(data.data["vol"].__len__()-20000-n_input, data.data["vol"].__len__()-n_input):
        if i%1000==0: print("Simulating i:",i)
        try:
            priceNow = (data.data["high"][i+n_input] + data.data["low"][i+n_input])/2

            # long
            if lastBuy:
                if priceNow>lastBuy*1.01 or priceNow<lastBuy*0.99:
                    winChange = (priceNow-lastBuy)*lastQuant
                    win += winChange
                    winChanges.append(winChanges)
                    print("made ",winChange,"at",priceNow,"now",win)
                    lastBuy = False
            # short
            else:
                inputData = list()
                j = 0
                while inputData.__len__() < n_input:
                    inputData.append(data.data["high"][i + j])
                    inputData.append(data.data["low"][i + j])
                    inputData.append(data.data["vol"][i + j])
                    j += 1
                if predictBuy(inputData):
                    print("Buy  at",priceNow)
                    lastBuy = priceNow
                    lastQuant = 1/priceNow
        except Exception as e:
            print(e)
    return winChanges
winchanges = runSimulation()
print()