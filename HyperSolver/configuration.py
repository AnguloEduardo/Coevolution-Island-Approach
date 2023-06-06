def to_float(lst):
    return [float(item) for item in lst.split()]


def read():
    # Mapping between parameter names and their corresponding conversion functions
    type_map = {
        "number_of_tournaments": int,
        "number_of_parents_to_select": int,
        "individuals_to_exchange": int,
        "number_of_islands": int,
        "population_size": int,
        "generations": int,
        "crossover_probability": to_float,
        "mutation_probability": to_float,
        "migration_probability": to_float,
        "features": str.split,
        "heuristics": str.split,
        "number_rules": int,
        "training_split": str,
        "training_set": str,
        "lb": int,
        "ub": int,
        "nm": int,
        "pc": int,
        "nc": int,
    }

    parameters = []

    with open('parameters.conf', 'r') as config_file:
        for line in config_file:
            line = line.rstrip("\n").split(' = ')
            conversion_func = type_map[line[0]]
            parameters.append(conversion_func(line[1]))

    parameters.append(parameters[3] * parameters[4])  # number_hh calculation

    return parameters

