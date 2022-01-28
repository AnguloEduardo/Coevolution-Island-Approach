import pandas as pd
from csv import writer

def generate_csv():
    df = pd.DataFrame({
        'Parameters': ['Population size', 'Number of generations', 'Crossover probability', 'Mutation probability',
                       'Backpack capacity', 'Maximum backpack weight', 'Instance used'],
        'Data': [population_size, generations, crossover_probability, mutation_probability, backpack_capacity,
                 max_weight, file_name],
        'Current generation': current_generation,
        'Best solution\nfound so far\nin island 1': solution_1,
        'Best solution\nfound so far\nin island 2': solution_2,
        'Best solution\nfound so far\nin island 3': solution_3,
        'Best solution\nfound so far\nin island 4': solution_4,
        'Best solution\nin island 1': ['Weight', 'Value', 'Backpack configuration', None, None, None, None],
        'Results 1': [best_individual[0].weight, best_individual[0].value, best_individual[0].chromosome,
                      None, None, None, None],
        'Best solution\nin island 2': ['Weight', 'Value', 'Backpack configuration', None, None, None, None],
        'Results 2': [best_individual[1].weight, best_individual[1].value, best_individual[1].chromosome,
                      None, None, None, None],
        'Best solution\nin island 3': ['Weight', 'Value', 'Backpack configuration', None, None, None, None],
        'Results 3': [best_individual[2].weight, best_individual[2].value, best_individual[2].chromosome,
                      None, None, None, None],
        'Best solution\nin island 4': ['Weight', 'Value', 'Backpack configuration', None, None, None, None],
        'Results 4': [best_individual[3].weight, best_individual[3].value, best_individual[3].chromosome,
                      None, None, None, None],
    })
    # Saving the data frame into a .csv file
    df.to_csv('coevolution.csv', index=False, encoding='utf_8_sig')


def write_data(data):
    with open('coevolution.csv', 'a', newline='') as f_object:
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(data)
        # Close the file object
        f_object.close()


def write_to_csv():
    empty = [None, None, None, None, None, None, None]
    list_data_1 = ['Population size', population_size, current_generation[0], solution_1[0], solution_2[0],
                   solution_3[0], solution_4[0], 'Weight', best_individual[0].weight, 'Weight',
                   best_individual[1].weight, 'Weight', best_individual[2].weight, 'Weight', best_individual[3].weight]
    list_data_2 = ['Number of generations', generations, current_generation[1], solution_1[1], solution_2[1],
                   solution_3[1], solution_4[1], 'Value', best_individual[0].value, 'Value', best_individual[1].value,
                   'Value', best_individual[2].value, 'Value', best_individual[3].value]
    list_data_3 = ['Crossover probability', crossover_probability, current_generation[2], solution_1[2], solution_2[2],
                   solution_3[2], solution_4[2], 'Backpack configuration', best_individual[0].chromosome,
                   'Backpack configuration', best_individual[1].chromosome, 'Backpack configuration',
                   best_individual[2].chromosome, 'Backpack configuration', best_individual[3].chromosome]
    list_data_4 = ['Mutation probability', mutation_probability, current_generation[3], solution_1[3], solution_2[3],
                   solution_3[3], solution_4[3], None, None, None, None, None, None, None, None]
    list_data_5 = ['Backpack capacity', backpack_capacity, current_generation[4], solution_1[4], solution_2[4],
                   solution_3[4], solution_4[4], None, None, None, None, None, None, None, None]
    list_data_6 = ['Maximum backpack weight', max_weight, current_generation[5], solution_1[5], solution_2[5],
                   solution_3[5], solution_4[5], None, None, None, None, None, None, None, None]
    list_data_7 = ['Instance used', file_name, current_generation[6], solution_1[6], solution_2[6],
                   solution_3[6], solution_4[6], None, None, None, None, None, None, None, None]
    write_data(empty)
    write_data(list_data_1)
    write_data(list_data_2)
    write_data(list_data_3)
    write_data(list_data_4)
    write_data(list_data_5)
    write_data(list_data_6)
    write_data(list_data_7)