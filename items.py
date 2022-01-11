class Items:
    def __init__(self, ID, value, weight):
        self.ID = ID
        self.value = value
        self.weight = weight

    def getID(self):
        return self.ID

    def getvalue(self):
        return self.value

    def getWeight(self):
        return self.weight

    def getvaluePerWeight(self):
        return self.value / self.weight
