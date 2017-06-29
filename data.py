import csv
import random

rows = list()
csvfile = open('btceUSD.csv')
reader = csv.reader(csvfile)

def getLine():
    global reader
    try:
        row = next(reader)
    except Exception as e:
        print("DATA SET ENDED!")
        csvfile = open('btceUSD.csv')
        reader = csv.reader(csvfile)
        offset = round(random.random()*1000) +1
        for i in range(offset): row = next(reader)

    timestamp = row[0]
    priceStr = row[1]
    price = float(priceStr)
    volume = float(row[2])
    return price

def getTrainingData(inputLength, futureLength):
    data = list()
    for i in range(inputLength+futureLength): data.append(getLine())

    input = data[:inputLength]
    futurePrice = sum(data[inputLength:])/futureLength

    if futurePrice > input[-1]: output = [1,0]
    else: output = [0,1]

    return (input, output)

def getTrainingBatch(batchSize, inputLength, futureLength):
    inputs = list()
    outputs = list()
    for i in range(batchSize):
        data = getTrainingData(inputLength, futureLength)
        inputs.append(data[0])
        outputs.append(data[1])
    return (inputs, outputs)

test = getTrainingBatch(5, 100,20)
print()