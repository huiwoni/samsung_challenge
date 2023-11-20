import csv
import json

input_file_name_1 = "test.csv"
output_file_name  = "test_jpg.txt"
with open(input_file_name_1, "r", encoding="utf-8", newline="") as input_file_1, \
        open(output_file_name, "w", encoding="utf-8", newline="") as output_file :


    reader_1 = csv.reader(input_file_1)

    col_names_1 = next(reader_1)

    output_file.write('img_name')

    for cols_1 in reader_1:

        cols_2 = str(str(cols_1[0]) + '.jpg' + '\n')

        output_file.write(cols_2)