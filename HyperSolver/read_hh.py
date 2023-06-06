from HyperHeuristic import HyperHeuristic
from ast import literal_eval


def read_hh(len_features, len_heuristics, nRules, hh_path):
    feature, heuristic, conditions, actions, rules, hh = [], [], [], [], [], []
    with open(hh_path + 'best_hh.txt', 'r') as results:
        text = results.read()
    tokens = iter(text.split())

    if next(tokens).rstrip(",:") == 'Features':
        for _ in range(len_features):
            feature.append(literal_eval(next(tokens)))
    if next(tokens).rstrip(",:") == 'Heuristics':
        for _ in range(len_heuristics):
            heuristic.append(literal_eval(next(tokens)))
    if next(tokens).rstrip(",:") == 'Rules':
        for _ in range(nRules):
            for _ in range(len_features):
                rules.append(float(literal_eval(next(tokens))))
            next(tokens)  # Token ignored
            actions.append(literal_eval(next(tokens)))
            conditions.append(list(rules))
            rules = []
        hh.append(HyperHeuristic(feature, heuristic, actions, conditions))
        next(tokens)
        next(tokens)
    return hh
