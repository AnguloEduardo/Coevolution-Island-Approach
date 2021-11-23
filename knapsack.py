from items import Items


class Knapsack:
    # Creates a new instance of Knapsack
    def __init__(self, capacity):
        self.capacity = capacity
        profit = 0
        items = list(Items())

    # Creates a new instance of Knapsack from an existing instance
    def knapsack(self, knapsack):
        self.capacity = knapsack.capacity
        self.profit = knapsack.profit
        self.items = knapsack.items

    # Returns the current capacity
    def getCapacity(self):
        return self.capacity

    # Returns the current profit
    def getProfit(self):
        return self.profit

    # Returns the number of items
    def getItems(self):
        return len(self.items)

    # Revises if the item provided can be packed in this knapsack
    def canPack(self, item):
        return item.getWeight() <= self.getCapacity()

    # Packs and item into this knapsack
    def pack(self, item):
        if item.getWeight() <= self.getCapacity():
            self.items.append(item)
            self.capacity -= item.getWeight()
            self.profit += item.getProfit()
            return True
        return False
