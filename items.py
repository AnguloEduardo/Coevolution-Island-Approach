class Items:
    def __init__(self, ID, profit, weight):
        self.ID = ID
        self.profit = profit
        self.weight = weight

    def getID(self):
        return self.ID

    def getProfit(self):
        return self.profit

    def getWeight(self):
        return self.weight

    def getProfitPerWeight(self):
        return self.profit / self.weight
