import csv

input_file = 'resultados_direct_solver_0.txt'
output_file = 'results_DS_0.csv'

delimiter = ' '

with open(input_file, 'r') as txt_file:
    with open(output_file, 'w', newline='') as csv_file:
        # Create a csv reader object for the input file
        txt_reader = csv.reader(txt_file, delimiter=delimiter)

        # Create a csv writer object for the output file
        csv_writer = csv.writer(csv_file)

        # Read each row from the input file and write it to the output file
        for row in txt_reader:
            csv_writer.writerow(row)
