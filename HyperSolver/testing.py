# Libraries
# =======================
import os
from read_hh import read_hh
from knapsack_HyperSolver import Knapsack
from knapsack_HyperSolver import read_instance
from tqdm import tqdm
from problem import ProblemCharacteristics


def test(features, heuristics, number_rules, number_hh, training_split, population_size, generations, number_of_islands,
         run_times, num_exp):
    problem_pool = []
    # Paths to the problem instance and to the solution folder
    root = os.getcwd()
    hh_path = root + '\\experiments\\ga\\Test set A\\' + training_split + '\\Training\\' + str(population_size) + '-' \
              + str(generations) + '-' + str(number_of_islands) + '-' + str(run_times) + '\\' + str(num_exp) + '\\'
    folder_instance = root + '\\Instances KP\\ga\\Test set A\\' + training_split + '\\Test'
    os.chdir(folder_instance)
    file_path, file_names = [], []
    results = open(hh_path + 'results' + '.txt', 'a')

    # Iterate over all the files in the directory
    for file in os.listdir():
        # Create the filepath of particular file
        file_names.append(f"{file}")
        file_path.append(f"{folder_instance}\\{file}")

    for file in file_path:
        problem_pool.append(ProblemCharacteristics(read_instance(file)))

    HHs = read_hh(len(features), len(heuristics), number_rules, number_hh, hh_path)

    for x in tqdm(range(len(HHs))):
        results = HHs[x].evaluate_testing(problem_pool, results)
        results.write("\n")

    results.close()
