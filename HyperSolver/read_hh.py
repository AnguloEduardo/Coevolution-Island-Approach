import os
from HyperHeuristic import HyperHeuristic


def read_hh(len_features, len_heuristics, nRules, len_hheuristics):
    feature, heuristic, conditions, actions, rules, hh = [], [], [], [], [], []
    # Paths to the problem instance and to the solution folder
    best_hhs = '\\experiments\\ga\\Test set (reduced)\\Training Set\\10-1000-4-30\\0'
    folder_solution = os.getcwd() + best_hhs
    os.chdir(folder_solution)
    file_path = []
    # Iterate over all the files in the directory
    for file in os.listdir():
        # Create the filepath of particular file
        file_path.append(f"{file}")

    for file in file_path:
        if file == 'Final Results.txt':
            file = open(file, "r")
            text = file.read()
            tokens = text.split()
            for _ in range(len_hheuristics):
                if tokens.pop(0).rstrip(",:") == 'Features':
                    for _ in range(len_features):
                        feature.append(tokens.pop(0).lstrip("[',").rstrip("',]"))
                if tokens.pop(0).rstrip(",:") == 'Heuristics':
                    for _ in range(len_heuristics):
                        heuristic.append(tokens.pop(0).lstrip("[',").rstrip("',]"))
                if tokens.pop(0).rstrip(",:") == 'Rules':
                    for _ in range(nRules):
                        for _ in range(len_features):
                            rules.append(float(tokens.pop(0).lstrip("[',").rstrip(",]")))
                        tokens.pop(0)  # Token ignored
                        actions.append(tokens.pop(0))
                        conditions.append(list(rules))
                        rules = []
                hh.append(HyperHeuristic(feature, heuristic, actions, conditions))
                feature, heuristic, conditions, actions, rules = [], [], [], [], []
    return hh
