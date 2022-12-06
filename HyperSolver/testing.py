# Libraries
# =======================
import os
from read_hh import read_hh
from knapsack_HyperSolver import read_instance
from tqdm import tqdm
from problem import ProblemCharacteristics


def test(number_of_islands, run_times, population_size, generations, features, heuristics, number_rules, training_split,
         number_hh, num_exp):
    # Paths to the problem instance and to the solution folder
    root = os.getcwd()
    hh_path = root + '\\experiments\\ga\\Test set A\\' + training_split + '\\Training\\' + str(population_size) + '-' \
              + str(generations) + '-' + str(number_of_islands) + '-' + str(run_times) + '\\' + str(num_exp) + '\\'

    folder_instance = root + '\\Instances KP\\ga\\Test set A\\' + training_split + '\\Test'
    os.chdir(folder_instance)

    # Iterate over all the files in the directory
    # Create the filepath of particular file
    file_path = [f"{folder_instance}\\{file}" for file in os.listdir()]
    problem_pool = [ProblemCharacteristics(read_instance(file)) for file in file_path]
    HHs = read_hh(len(features), len(heuristics), number_rules, number_hh, hh_path)

    for x in tqdm(range(len(HHs))):
        HHs[x].evaluate_testing(problem_pool, hh_path)
