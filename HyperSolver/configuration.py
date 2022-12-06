def to_float(lst):
    for index, item in enumerate(lst):
        lst[index] = float(item)
    return lst


def read():
    # Control variables
    config_file = open('parameters.conf', 'r')

    line = config_file.readline().rstrip("\n").split(' = ')
    number_of_tournaments = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    number_of_parents_to_select = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    individuals_to_exchange = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    number_of_islands = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    run_times = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    population_size = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    generations = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    crossover_probability = to_float(line[1].split())

    line = config_file.readline().rstrip("\n").split(' = ')
    mutation_probability = to_float(line[1].split())

    line = config_file.readline().rstrip("\n").split(' = ')
    migration_probability = to_float(line[1].split())

    line = config_file.readline().rstrip("\n").split(' = ')
    features = line[1].split()

    line = config_file.readline().rstrip("\n").split(' = ')
    heuristics = line[1].split()

    line = config_file.readline().rstrip("\n").split(' = ')
    number_rules = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    training_split = line[1]

    line = config_file.readline().rstrip("\n").split(' = ')
    lb = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    ub = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    nm = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    pc = int(line[1])

    line = config_file.readline().rstrip("\n").split(' = ')
    nc = int(line[1])

    number_hh = number_of_islands * population_size

    parameters = [number_of_tournaments, number_of_parents_to_select, individuals_to_exchange, number_of_islands,
                  run_times, population_size, generations, crossover_probability, mutation_probability,
                  migration_probability, features, heuristics, number_rules, training_split, number_hh, lb, ub, nm, pc,
                  nc]

    return parameters
