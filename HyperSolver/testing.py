# Libraries
# =======================
import os
from read_hh import read_hh
from knapsack_HyperSolver import read_instance
from tqdm import tqdm
from problem import ProblemCharacteristics


def test(number_of_islands, population_size, generations, features, heuristics, number_rules, training_split,
         training_set, num_exp):
    # Paths to the problem instance and to the solution folder
    root = os.getcwd()
    hh_path = os.path.join(root, 'experiments', 'ga', training_set, training_split + str(population_size) + '-' +
                           str(generations) + '-' + str(number_of_islands), str(num_exp))

    folder_instance = os.path.join(root, 'Instances KP', 'ga', training_set, training_split, 'Test')
    os.chdir(folder_instance)

    # Iterate over all the files in the directory
    # Create the filepath of particular file
    file_paths = [os.path.join(folder_instance, file) for file in os.listdir(folder_instance)]
    problem_pool = [ProblemCharacteristics(read_instance(file)) for file in file_paths]

    # Read HH
    HHs = read_hh(len(features), len(heuristics), number_rules, hh_path)

    # Solve the testing set
    for HH in tqdm(HHs):
        HH.evaluate_testing(problem_pool, hh_path)
