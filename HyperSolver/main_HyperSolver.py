import sys
from training import train
from testing import test
from configuration import read

if __name__ == "__main__":
    parameter_names = [
        "number_of_tournaments", "number_of_parents_to_select", "individuals_to_exchange",
        "number_of_islands", "population_size", "generations", "crossover_probability",
        "mutation_probability", "migration_probability", "features", "heuristics",
        "number_rules", "training_split", "training_set", "number_hh", "lb", "ub", "nm", "pc", "nc"
    ]

    # Create a dictionary for parameters
    parameters = dict(zip(parameter_names, read()))

    # Check the number of command-line arguments
    if len(sys.argv) < 2:
        print('Error: Missing command-line argument. Please specify "train" or "test".')
        sys.exit(1)

    if sys.argv[1] == 'train':
        print('Training')
        train_params = {k: parameters[k] for k in ("number_of_tournaments", "number_of_parents_to_select",
                                                   "individuals_to_exchange", "number_of_islands", "population_size",
                                                   "generations", "crossover_probability", "mutation_probability",
                                                   "migration_probability", "features", "heuristics", "number_rules",
                                                   "training_split", "training_set", "lb", "ub", "nm", "pc", "nc")}
        train(**train_params)

    elif sys.argv[1] == 'test':
        if len(sys.argv) < 3:
            print('Error: Missing experiment number for test.')
            sys.exit(1)
        experiment_num = sys.argv[2]
        print('Testing')
        test_params = {k: parameters[k] for k in ("number_of_islands", "population_size", "generations",
                                                  "features", "heuristics", "number_rules", "training_split",
                                                  "training_set")}
        test_params['experiment_num'] = sys.argv[2]
        test(**test_params)
    else:
        print('Error: Invalid command-line argument. Please specify "train" or "test".')
