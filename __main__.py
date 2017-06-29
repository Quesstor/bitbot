import csv
trades = list()
class macd:
    def __init__(self, length1, length2, buyFunction, sellFunction):
        if length1 < length2:
            self.lshort = length1
            self.llong = length2
        else:
            self.lshort = length2
            self.llong = length1

        self.priceHistory = list()
        self.macd = 0
        self.m1 = 0
        self.m2 = 0

        self.buyFunction = buyFunction
        self.sellFunction = sellFunction

    def calc(self):
        if self.priceHistory.__len__() < self.llong: return
        self.m1 = sum(self.priceHistory[-self.lshort:]) / self.lshort
        self.m2 = sum(self.priceHistory[-self.llong:]) / self.llong
        newmacd = self.m2 - self.m1

        threshhold = self.m2 / 100

        if self.macd < 0 and newmacd > 0:
            self.buyFunction(self.priceHistory[-1])
        elif self.macd > 0 and newmacd < 0:
            self.sellFunction(self.priceHistory[-1])
        self.macd = newmacd

    def addTrade(self, price):
        if self.priceHistory.__len__() > self.llong:
            self.priceHistory = self.priceHistory[1:]
        self.priceHistory.append(price)
        self.calc()


class vendor:
    def __init__(self):
        self.cash = 0
        self.fee = 0.002

    def buy(self, p):
        #print("BUY : {:10.4f}".format(p) + " - {:10.4f}".format(self.cash))
        self.cash -= p
        self.payFee(p)

    def sell(self, p):
        #print("SELL: {:10.4f}".format(p) + " - {:10.4f}".format(self.cash))
        self.cash += p
        self.payFee(p)

    def payFee(self, p):
        self.cash -= p * self.fee


me = vendor()
macd = macd(5, 40, me.buy, me.sell)

i=0
with open('btceUSD.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        timestamp = row[0]
        priceStr = row[1]
        price = float(priceStr)
        volume = float(row[2])

        macd.addTrade(price)

        i += 1
        if i % 100000 == 0: print(str(me.cash) + " - " +str(round(price)))

print(str(me.cash))