import csv
import random
import math
import numpy

csvfile = open('.btceEur.csv')  # 0:timestamp, 1:price, 2:volume
reader = csv.reader(csvfile)
rawLines = [r for r in reader]
data = dict()
data["high"] = list()
data["low"] = list()
data["vol"] = list()

currentLineIndex = 0
def getTimePeriodData(startTimestamp, stopTimestamp):
    global currentLineIndex
    timePeriodData = list()
    line = rawLines[currentLineIndex]
    timestamp = int(line[0])
    while timestamp >= startTimestamp and timestamp<stopTimestamp:
        currentLineIndex += 1
        timePeriodData.append((int(line[0]), float(line[1]), float(line[2])))
        line = rawLines[currentLineIndex]
        timestamp = int(line[0])

    return timePeriodData

def prepareData():
    global data
    last = 0
    secondsPerTimePeriod = 60
    timestamp = int(rawLines[0][0])
    while 1:
        try:
            timePeriodData = getTimePeriodData(timestamp, timestamp+secondsPerTimePeriod)
            if timePeriodData.__len__()>0:
                prices = [d[1] for d in timePeriodData]
                volume = [d[2] for d in timePeriodData]
                high = max(prices)
                low = min(prices)
                vol = sum(volume)
                last = prices[-1]
            else:
                high = last
                low = last
                vol = 0
            data["high"].append(high)
            data["low"].append(low)
            data["vol"].append(vol)
            timestamp += secondsPerTimePeriod
        except Exception as e:
            dataLength = data["vol"].__len__()
            print("File read to end. Data length = ", data["vol"].__len__())
            # import matplotlib.pyplot as plt
            # plt.plot(range(dataLength), data["high"], 'r--',range(dataLength), data["low"], 'b--', range(dataLength), data["vol"], 'g^')
            # plt.show()
            return
prepareData()

offset = 0
cases = 20
ouputcounts = dict()
for i in range(cases): ouputcounts[i] = 0

def getTrainingData():
    global data, offset
    minutes = 1
    hours = 60

    inputDataLength = 24*hours
    futureDataLength = 90*minutes

    # offset += 100
    # if offset>data["vol"].__len__()-inputDataLength-futureDataLength:
    #     offset = 0

    offset=round(random.random() * (data["vol"].__len__() - inputDataLength-futureDataLength))

    inputStart = offset
    inputEnd = offset+inputDataLength
    futureStart = offset+inputDataLength
    futureEnd = offset+inputDataLength+futureDataLength

    buyPrice = (sum(data["high"][inputEnd-50:inputEnd])+sum(data["low"][inputEnd-50:inputEnd]))/100
    futureMeans = list()
    for i in range(futureStart, futureEnd):
        futureMeans.append((data["high"][i]+data["low"][i])/2)
    futureMax = max(futureMeans)
    futureMin = min(futureMeans)

    # increase = futureMax / buyPrice - 1
    # decrease = abs(futureMin / buyPrice - 1)

    futureMean = numpy.mean(futureMeans)
    change = futureMean / buyPrice -1
    change = change*1000+cases/2
    index = int(max(0,min(cases-1,round(change))))

    output = list()
    for i in range(cases):
        if i==index: output.append(1)
        else: output.append(0)
    ouputcounts[index] += 1

    inputData = list()
    for i in range(inputStart, inputEnd):
        inputData.append(data["high"][i])
        inputData.append(data["low"][i])
        inputData.append(data["vol"][i])
    return (inputData, output)

def getTrainingBatch(batchSize):
    inputs = list()
    outputs = list()
    for i in range(batchSize):
        data = getTrainingData()
        inputs.append(data[0])
        outputs.append(data[1])
    return (inputs, outputs)
