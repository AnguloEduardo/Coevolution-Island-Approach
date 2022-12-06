import numpy as np
import math


def split(a, n):
    x = []
    k, m = divmod(a, n)
    for i in range(n):
        x.append(int(i * k + min(i, m)))
    return x


class HyperHeuristic:
    # Constructor first case
    #   0. features = A list with the names of the features to be used by this hyper-heuristic
    #   1. heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   2. nbRules = The number of rules to be contained in this hyper-heuristic
    #   3. length of problem pool = The number of problems to train the model
    #   4. lower bound of the rules
    #   5. upper bound of the rules

    # Constructor second case
    #   0. features = A list with the names of the features to be used by this hyper-heuristic
    #   1. heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   2. actions = A list indicating which heuristic triggers with each rule
    #   3. conditions = A list of rules composed of features
    #   4. max_weight
    #   5. capacity

    def __init__(self, *args):
        if len(args) == 6:
            self.features = args[0].copy()
            self.heuristics = args[1].copy()
            self.problems_solved = [0] * args[3]  # number of problems to solve
            self.fitness = int(0)
            self.actions = []
            self.conditions = []
            for i in range(args[2]):
                # Initializing conditions randomly
                self.conditions.append([0] * len(args[0]))
                self.conditions[i] = np.random.uniform(args[4], args[5], (len(args[0]),))

                # Adding the heuristics to use (in this case we use 5 heuristics)
                # MAX_PROFIT, MAX_PROFIT/WEIGHT, MIN_WEIGHT, MARKOVITZ, DEFAULT
                # We divide the number of rules into five spaces. If the number of rules
                # is a multiple of five, the spaces will have the same length. In the case
                # of an odd number, some spaces will be smaller than the others.
                ranges = split(args[2], 5)
                if i < ranges[1]:
                    self.actions.append(args[1][0])
                elif i < ranges[2]:
                    self.actions.append(args[1][1])
                elif i < ranges[3]:
                    self.actions.append(args[1][2])
                elif i < ranges[4]:
                    self.actions.append(args[1][3])
                else:
                    self.actions.append(args[1][4])

        elif len(args) == 4:
            self.features = args[0].copy()
            self.heuristics = args[1].copy()
            self.actions = args[2].copy()
            self.conditions = args[3].copy()

        else:
            self.features = args[0].copy()
            self.heuristics = args[1].copy()
            self.actions = args[2].copy()
            self.conditions = args[3].copy()
            self.problems_solved = [0] * args[4]  # number of problems to solve
            self.fitness = int(0)

    # Returns the next heuristic to use
    def next_heuristic(self, problem):
        min_distance = float("inf")
        index = -1
        state = []
        for i in range(len(self.features)):
            state.append(
                problem.individual.get_feature(self.features[i], problem.get_total_weight(), problem.get_total_value(),
                                               problem.get_items()))
        for i in range(len(self.conditions)):
            distance = self.__distance(self.conditions[i], state)
            if distance < min_distance:
                min_distance = distance
                index = i
        heuristic = self.actions[index]
        return heuristic

    # Returns the string representation of this dummy hyper-heuristic
    def __str__(self):
        text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n"
        for i in range(len(self.conditions)):
            text += "\t" + str(self.conditions[i]) + " => " + self.actions[i] + "\n"
        return text

    # Returns the Euclidean distance between two vectors
    def __distance(self, vector_a, vector_b):
        distance = 0
        for i in range(len(vector_a)):
            distance += (vector_a[i] - vector_b[i]) ** 2
        distance = math.sqrt(distance)
        return distance

    def evaluate(self, problem_pool):
        for idx, problem in enumerate(problem_pool):
            item = problem.individual.solve(self.next_heuristic(problem), problem)
            while item is not None:
                problem.individual.pack(item)
                item = problem.individual.solve(self.next_heuristic(problem), problem)
            if problem.update_fitness():
                self.problems_solved[idx] = int(1)
            else:
                self.problems_solved[idx] = int(0)
            problem.reset()
        self.fitness = sum(self.problems_solved)

    def evaluate_testing(self, problem_pool, hh_path):
        results = open(hh_path + 'results' + '.txt', 'a')
        for idx, problem in enumerate(problem_pool):
            item = problem.individual.solve(self.next_heuristic(problem), problem)
            while item is not None:
                problem.individual.pack(item)
                item = problem.individual.solve(self.next_heuristic(problem), problem)
            results.write(str(problem.individual.get_value()) + " ")
            problem.reset()
        results.write('\n')
        results.close()

    def get_fitness(self):
        return self.fitness

    def get_conditions(self):
        return self.conditions

    def get_actions(self):
        return self.actions
