import json
import csv

input_file_1_1 = 'annotation_train.csv'
input_file_1_2 = 'annotation_train_image.csv'
output_file = 'captions_train2014.json'


with open(input_file_1_1, 'r', encoding='utf-8') as input_file_1, \
        open(input_file_1_2, 'r', encoding='utf-8') as input_file_2, \
            open(output_file, 'w', encoding='utf-8') as output_file:

    reader_1 = csv.reader(input_file_1)
    reader_2 = csv.reader(input_file_2)

    col_names_1 = next(reader_1)
    col_names_2 = next(reader_2)

    images = []
    annotations =[]


    for reader1 in reader_2:
        images.append({'file_name': str(reader1[0] + '.jpg'), 'id' : str(reader1[0]), 'height' : 1, 'width' : 1})

    for reader2 in reader_1:
        annotations.append({'image_id': str(reader2[0]), 'id' : str(reader2[1]), 'caption' : str(reader2[2])})


    file_out = {'images' : images, 'annotations' : annotations}
    json.dump(file_out, output_file)

input_file_2_1 = 'annotation_val.csv'
input_file_2_2 = 'annotation_val_image.csv'
output_file = 'captions_val2014.json'

with open(input_file_2_1, 'r', encoding='utf-8') as input_file_1, \
        open(input_file_2_2, 'r', encoding='utf-8') as input_file_2, \
        open(output_file, 'w', encoding='utf-8') as output_file:

    reader_1 = csv.reader(input_file_1)
    reader_2 = csv.reader(input_file_2)

    col_names_1 = next(reader_1)
    col_names_2 = next(reader_2)

    images = []
    annotations = []

    for reader1 in reader_2:
        images.append({'file_name': str(reader1[0] + '.jpg'), 'id': str(reader1[0]), 'height': 1, 'width': 1})

    for reader2 in reader_1:
        annotations.append({'image_id': str(reader2[0]), 'id': str(reader2[1]), 'caption': str(reader2[2])})

    file_out = {'images': images, 'annotations': annotations}
    json.dump(file_out, output_file)

input_file_3_1 = 'test.csv'
output_file = 'image_info_test2014.json'

with open(input_file_3_1, 'r', encoding='utf-8') as input_file_1, \
        open(output_file, 'w', encoding='utf-8') as output_file:

    reader_1 = csv.reader(input_file_1)

    col_names_1 = next(reader_1)

    images = []
    annotations = []

    for reader1 in reader_1:
        images.append({'file_name': str(reader1[0] + '.jpg'), 'id': str(reader1[0]), 'height': 1, 'width': 1})


    file_out = {'images': images}
    json.dump(file_out, output_file)

