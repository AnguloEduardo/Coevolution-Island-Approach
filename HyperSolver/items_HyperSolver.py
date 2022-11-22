class Items:
    def __init__(self, ID, value, weight):
        self.ID = ID
        self.value = int(value)
        self.weight = float(weight)

    def get_id(self):
        return self.ID

    def get_value(self):
        return self.value

    def get_weight(self):
        return self.weight

    def get_value_per_weight(self):
        return self.value / self.weight
