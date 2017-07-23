import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import csv
import matplotlib.pyplot as plt
import math

def show(inputData, expectedData, predictedData):
    x1 = range(len(inputData))
    x2 = range(len(inputData), len(inputData)+len(expectedData))

    plt.clf()
    plt.setp(plt.plot(x1, inputData, "b--"), linewidth=0.5)
    plt.setp(plt.plot(x2, expectedData, "g--"), linewidth=0.5)
    plt.setp(plt.plot(x2, predictedData, "r--"), linewidth=0.5)
    plt.show()

csvfile = open('.bitstampUSD.csv')  # 0:timestamp, 1:price, 2:volume
reader = csv.reader(csvfile)
def getBatches(batchCount, n_input, n_output):
    global reader
    batches = list()
    rawlines = [float(next(reader)[1]) for i in range(batchCount+n_input+n_output)]
    offset = 0
    for b in range(batchCount):
        input = rawlines[offset:offset+n_input]
        output = rawlines[offset+n_input:offset+n_input+n_output]

        #normalize
        m = max(input)
        input = [i/m for i in input]
        output = [o/m for o in output]

        batch = input, output
        batches.append(batch)
        offset+=1
    return batches

def getTestBatches(batchCount, n_input, n_output):
    batches = list()
    for b in range(batchCount):
        data = [math.sin(i*0.01) for i in range(n_input+n_output)]
        batch = data[:n_input], data[n_input:]
        batches.append(batch)
    return batches

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
    layer_1 = tf.add(tf.matmul(input, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



n_input = 1000
n_output = 200
inputLayer = tf.placeholder(tf.float32, [None, n_input])
trainLayer = tf.placeholder(tf.float32, [None, n_output])
outputLayer = multilayer_perceptron(inputLayer, n_input, n_input, n_input, n_output)

loss = tf.reduce_sum(tf.square(outputLayer - trainLayer))
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

c = 0
while 1:
    print("loading batches")
    batches = getBatches(10000, n_input, n_output)

    print("training")
    for b in batches:
        x,y = b
        inputData = [x]
        trainData = [y]
        _, l = sess.run([train_step,loss], feed_dict={inputLayer: inputData, trainLayer: trainData})
        c+=1
        if c%1000 == 0:
            print("Step ", c, " Loss ", l)
            if c % 5000 == 0:
                x, y = getBatches(1, n_input, n_output)[0]
                predictedData = sess.run(outputLayer, feed_dict={inputLayer: [x]})
                predictedData = [x for x in predictedData[0]]
                show(x,y,predictedData)

print()