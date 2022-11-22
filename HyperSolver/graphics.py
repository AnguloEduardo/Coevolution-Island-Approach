import matplotlib.pyplot as plt
import numpy as np
import os

data = open(os.getcwd() + '\\experiments\\ga\\Test set A\\70-30\\Training\\10-1000-4-30\\0\\hh_results.txt', 'r')
folder_instance = os.getcwd() + '\\Instances KP\\ga\\Test set A\\70-30\\Test\\'
temp = []


if __name__ == '__main__':
    root = os.getcwd()
    os.chdir(folder_instance)
    file_path = []
    # Iterate over all the files in the directory
    for file in os.listdir():
        # Create the filepath of particular file
        file_path.append(f"{folder_instance}\\{file}")
    for file in file_path:
        lines = data.readline().rstrip("\n")
        line = lines.split(" ")
        line.pop()
        line = [eval(i) for i in line]
        file = file.split("\\")
        file = file[len(file)-1]
        plt.figure(figsize=(10, 7))
        plt.title('Instance ' + file)
        plt.xlabel('HyperHeuristics')
        plt.ylabel('Value')
        plt.plot(line)
        plt.savefig(root + '\\experiments\\graphs\\70-30\\' + file + '.jpg')
