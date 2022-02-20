import matplotlib.pyplot as plt
import numpy as np

formatted_data = open('experiments\\data.txt', 'r')
temp = []
file_name = ['ks_4_0', 'ks_19_0', 'ks_30_0', 'ks_40_0', 'ks_45_0', 'ks_50_0', 'ks_50_1',
             'ks_60_0', 'ks_82_0', 'ks_100_0', 'ks_100_1', 'ks_100_2', 'ks_106_0', 'ks_200_0', 'ks_200_1',
             'ks_300_0', 'ks_400_0', 'ks_500_0', 'ks_1000_0', 'ks_10000_0']


def reading():
    global temp
    lines = formatted_data.readline().rstrip("\n")
    line = lines.split(" ")
    islands = np.array(list(map(int, line[:30])))
    island_1 = np.array(list(map(int, line[30:60])))
    island_2 = np.array(list(map(int,line[60:90])))
    island_3 = np.array(list(map(int,line[90:120])))
    island_4 = np.array(list(map(int,line[120:])))
    temp = [islands, island_1, island_2, island_3, island_4]


if __name__ == '__main__':
    for i in range(20):
        reading()
        plt.figure(figsize=(10, 7))
        plt.title('Instance ' + file_name[i])
        plt.xlabel('Islands')
        plt.ylabel('Value')
        plt.boxplot(temp)
        plt.savefig('experiments\\graphs\\' + file_name[i])
