import csv
import random
import math

rows = list()
csvfile = open('.btceEUR.csv')
reader = csv.reader(csvfile)
print("reading file ...")
prices = [float(row[1]) for row in reader]
len = prices.__len__()
print("done")


def getTrainingData():
    inputDataLength = 10000
    futureDataLength = 100
    randomOffset = round(random.random() * (len - (inputDataLength+futureDataLength)))

    inputData = list()
    futureData = list()
    linecount = 0
    maxi = 0
    for price in prices[randomOffset : randomOffset+inputDataLength+futureDataLength]:
        if linecount<inputDataLength:
            if price > maxi: maxi = price
            inputData.append(price)
        else: futureData.append(price)
        linecount += 1

    buyPrice = inputData[-1]
    output = [0,1] #do nothing
    if min(futureData) > buyPrice*0.99 and max(futureData) > buyPrice * 1.01: output=[1,0] #buy

    inputData = [i/maxi for i in inputData]
    return (inputData, output)

def getTrainingBatch(batchSize):
    inputs = list()
    outputs = list()
    for i in range(batchSize):
        data = getTrainingData()
        inputs.append(data[0])
        outputs.append(data[1])
    return (inputs, outputs)
