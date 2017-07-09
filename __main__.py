import csv
import KI

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
i=0
with open('btceUSD.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        timestamp = row[0]
        priceStr = row[1]
        price = float(priceStr)
        volume = float(row[2])


        i += 1
        if i % 100000 == 0: print(str(me.cash) + " - " +str(round(price)))

print(str(me.cash))