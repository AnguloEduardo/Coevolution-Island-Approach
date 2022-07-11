import os
import statistics

temp = []
mean_temp = []
mean = []
data = open(os.getcwd() + '\\experiments\\ga\\Test set (reduced)\\300-50-4-30\\1\\table.txt', 'r')
text = data.read()
tokens = text.split()
for x in range(50):
    for _ in range(9):
        for y in range(30):
            temp.append(int(tokens.pop(0)))

        mean_temp.append(statistics.median_high(temp))
        temp = []
    mean.append(list(mean_temp))
    mean_temp = []

for x in range(50):
    print("{0:8.1f} \t {1:8.1f} \t {2:8.1f} \t {3:8.1f} \t {4:8.1f} \t {5:8.1f} \t {6:8.1f} \t"
          "{7:8.1f} \t {8:8.1f} \t".format(mean[x][0], mean[x][1], mean[x][2], mean[x][3], mean[x][4], mean[x][5],
                                           mean[x][6], mean[x][7], mean[x][8]))
