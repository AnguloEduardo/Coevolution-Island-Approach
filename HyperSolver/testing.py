# Libraries
# =======================
import os
from read_hh import read_hh
from knapsack_HyperSolver import Knapsack
from items_HyperSolver import Items
from tqdm import tqdm

# Paths to the problem instance and to the solution folder
root = os.getcwd()
instance = '\\Instances KP\\ga\\Test set A\\30-70\\Test'
folder_solution = '\\experiments\\ga\\Test set A\\30-70\\'
folder_instance = root + instance
os.chdir(folder_instance)
file_path, file_names = [], []
results = open(root + folder_solution + 'hh_results' + '.txt', 'a')

# Iterate over all the files in the directory
for file in os.listdir():
    # Create the filepath of particular file
    file_names.append(f"{file}")
    file_path.append(f"{folder_instance}\\{file}")

for kp in range(len(file_path)):
    os.chdir(root)
    # List with the items of the problem
    list_items = []
    # Reading files with the instance problem
    kp_instance = open(file_path[kp], 'r')
    problemCharacteristics = kp_instance.readline().rstrip("\n")
    problemCharacteristics = problemCharacteristics.split(", ")

    # This information needs to be taken from the .txt files
    # First element in the first row indicates the number of items
    # second element of the first row indicates the backpack capacity
    # from the second row and forth, the first element represent the profit
    # the second element represent the weight
    backpack_capacity = int(problemCharacteristics[0])  # Number of items in the problem
    max_weight = float(problemCharacteristics[1])  # Maximum weight for the backpack to carry

    # Creation of item's characteristics with the information from the .txt file
    for idx in range(backpack_capacity):
        instanceItem = kp_instance.readline().rstrip("\n")
        instanceItem = instanceItem.split(", ")
        list_items.append(Items(idx, float(instanceItem[1]), int(instanceItem[0])))
    kp_instance.close()

    HHs = read_hh(len(features), len(heuristics), number_rules, 30)

    for x in tqdm(range(len(HHs))):
        HHs[x].individual = Knapsack(max_weight, backpack_capacity)
        HHs[x].evaluate(list_items, max_weight)
        results.write(str(HHs[x].individual.getValue()) + " ")
    results.write("\n")
results.close()
