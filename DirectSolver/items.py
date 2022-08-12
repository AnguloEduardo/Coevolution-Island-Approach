class Items:
    def __init__(self, ID, value, weight):
        self.ID = ID
        self.value = int(value)
        self.weight = float(weight)

    def getID(self):
        return self.ID

    def getValue(self):
        return self.value

    def getWeight(self):
        return self.weight

    def getvaluePerWeight(self):
        return self.value / self.weight
