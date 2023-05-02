import sys
from training import train
from testing import test
from configuration import read


if __name__ == "__main__":
    '''
    parameters = [number_of_tournaments, number_of_parents_to_select, individuals_to_exchange, number_of_islands, 
                  run_times, population_size, generations, crossover_probability, mutation_probability,
                  migration_probability, features, heuristics, number_rules, training_split, number_hh, lb, up]
    '''
    parameters = read()
    if sys.argv[1] == 'train':
        print('Training')
        '''
        num_tournament, num_parents_to_select, individuals_to_exchange, number_islands, run_times, population_size, generations
        crossover_probability, mutation_probability, migration_probability, features, heuristics, number_rules, training_split
        lower_bound, upper_bound, nm, pc, nc
        '''
        train(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6],
              parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13],
              parameters[15], parameters[16], parameters[17], parameters[18], parameters[19])
    elif sys.argv[1] == 'test':
        experiment_num = sys.argv[2]
        print('Testing')
        '''
        number_of_islands, run_times, population_size, generations,features, heuristics, number_rules,
        training_split, number_hh, experiment_num
        '''
        test(parameters[3], parameters[4], parameters[5], parameters[6], parameters[10], parameters[11], parameters[12],
             parameters[13], parameters[14], experiment_num)
    else:
        print('Neither train nor test were selected')
