from knapsack_HyperSolver import Knapsack


class ProblemCharacteristics:
    def __init__(self, *args):
        self.items = args[0][0].copy()
        self.max_weight = int(args[0][1])
        self.backpack_capacity = int(args[0][2])
        self.fitness = int(0)
        self.individual = Knapsack(self.get_max_weight(), len(self.get_items()))
        self.total_weight = float(args[0][3])
        self.total_value = int(args[0][4])

    def get_items(self):
        return self.items

    def get_max_weight(self):
        return self.max_weight

    def get_backpack_capacity(self):
        return self.backpack_capacity

    def get_total_weight(self):
        return self.total_weight

    def get_total_value(self):
        return self.total_value

    def update_fitness(self):
        if self.individual.get_value() < self.fitness:
            return self.individual.get_value() / self.fitness
        else:
            self.fitness = self.individual.get_value()
            return float(1)

    def reset(self):
        self.individual.totalWeight = self.max_weight
        self.individual.value = int(0)
        self.individual.chromosome = [0] * len(self.items)
