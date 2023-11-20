import csv
import json

input_file_name_1 = "train.csv"
output_file_name  = "train_jpg.txt"
with open(input_file_name_1, "r", encoding="utf-8", newline="") as input_file_1, \
        open(output_file_name, "w", encoding="utf-8", newline="") as output_file :

    reader_1 = csv.reader(input_file_1)

    col_names_1 = next(reader_1)
    i=0

    for cols_1 in reader_1:
        i+=1
        cols_2 = str(i) + ' ' + str(str(cols_1[0]) + '.jpg') + ' ' + str(cols_1[1])+"\n"

        output_file.write(cols_2)