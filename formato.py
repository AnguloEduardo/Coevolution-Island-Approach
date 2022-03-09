file_name = ['ks_4_0', 'ks_19_0', 'ks_30_0', 'ks_40_0', 'ks_45_0', 'ks_50_0', 'ks_50_1',
             'ks_60_0', 'ks_82_0', 'ks_100_0', 'ks_100_1', 'ks_100_2', 'ks_106_0', 'ks_200_0', 'ks_200_1',
             'ks_300_0', 'ks_400_0', 'ks_500_0', 'ks_1000_0', 'ks_10000_0']
format_data = open('experiments\\data.txt', 'a')
number_islands = 2

def formatting():
    for _ in range(30):
        for i in range(5):
            next(data_simple)
        line = data_simple.readline().rstrip("\n")
        print(line)
        line = line.split(": ")
        format_data.write(line[1] + " ")

if __name__ == '__main__':
    format_data.write("\n")
    data_simple = open('experiments\\' + file_name + '\\' + file_name + '_' + str(number_islands) + '_Islands_simple.txt', 'r')
    formatting()
    data_simple.close()
    format_data.close()