import json
import csv

input_file_1 = 'annotation_val.csv'
input_file_2 = 'image_info_test2014.json'
output_file = 'instances_val2014.json'

#
with open(input_file_1, 'r', encoding='utf-8') as input_file_1, \
        open(input_file_2, 'r', encoding='utf-8') as input_file_2, \
            open(output_file, 'w', encoding='utf-8') as output_file:

    data = json.load(input_file_2)
    reader_1 = csv.reader(input_file_1)
    col_names_1 = next(reader_1)
    file = []

    for reader in reader_1:
        file.append({'file_name': str(reader[0] + '.jpg'), 'id': str(reader[0]), 'height': 1, 'width': 1})

    file_out = {'images' : file, 'categories' : data['categories']}
    json.dump(file_out, output_file)

