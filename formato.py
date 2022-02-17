file_name = 'ks_10000_0'
format_data = open('experiments\\data.txt', 'a')

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
    data_simple = open('experiments\\' + file_name + '\\' + file_name + '_4_Islands_simple.txt', 'r')
    formatting()
    data_simple = open('experiments\\' + file_name + '\\' + file_name + '_Island_1_simple.txt', 'r')
    formatting()
    data_simple = open('experiments\\' + file_name + '\\' + file_name + '_Island_2_simple.txt', 'r')
    formatting()
    data_simple = open('experiments\\' + file_name + '\\' + file_name + '_Island_3_simple.txt', 'r')
    formatting()
    data_simple = open('experiments\\' + file_name + '\\' + file_name + '_Island_4_simple.txt', 'r')
    formatting()
    data_simple.close()
    format_data.close()