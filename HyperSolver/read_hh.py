from HyperHeuristic import HyperHeuristic


def read_hh(len_features, len_heuristics, nRules, len_hheuristics, hh_path):
    feature, heuristic, conditions, actions, rules, hh = [], [], [], [], [], []
    results = open(hh_path + 'hh_results.txt', 'r')
    text = results.read()
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
