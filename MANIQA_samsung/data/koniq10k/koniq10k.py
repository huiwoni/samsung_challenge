import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import csv


class Koniq10k(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Koniq10k, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []

 #       with open(self.txt_file_name, 'r') as listFile:

        ########################################################### append
        with open(txt_file_name, 'r') as col_names:
            listFile = csv.reader(col_names)
            reader_1 = next(listFile)
        ###########################################################
            for line in listFile:
                #################################append
                dis, score = line[0], line[1]
                #################################
                # dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)
                elif dis[2:] in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range


    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]

        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)

        return sample



class Koniq10k_HZ(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, dis_HZ_path, keep_ratio):
        super(Koniq10k_HZ, self).__init__()
        self.dis_path = dis_path
        self.dis_HZ_path = dis_HZ_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []

        #       with open(self.txt_file_name, 'r') as listFile:

        ########################################################### append
        with open(txt_file_name, 'r') as col_names:
            listFile = csv.reader(col_names)
            reader_1 = next(listFile)
            ###########################################################
            for line in listFile:
                #################################append
                dis, score = line[0], line[1]
                #################################
                # dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img_HZ = cv2.imread(os.path.join(self.dis_HZ_path, d_img_name), cv2.IMREAD_COLOR)
        d_img_HZ = cv2.resize(d_img_HZ, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img_HZ = cv2.cvtColor(d_img_HZ, cv2.COLOR_BGR2RGB)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img_HZ = np.array(d_img_HZ).astype('float32') / 255
        d_img = np.array(d_img).astype('float32') / 255
        d_img_HZ = np.transpose(d_img_HZ, (2, 0, 1))
        d_img = np.transpose(d_img, (2, 0, 1))

        d_img_RGB_HZ = np.append(d_img, d_img_HZ, axis=0)

        score = self.data_dict['score_list'][idx]

        sample = {
            'd_img_org': d_img_RGB_HZ,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

class Koniq10k_inference(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Koniq10k_inference, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        #       with open(self.txt_file_name, 'r') as listFile:
        ########################################################### append
        with open(txt_file_name, 'r') as col_names:
            listFile = csv.reader(col_names)
            reader_1 = next(listFile)
            ###########################################################
            for line in listFile:
                #################################append
                dis = line[0]
                #################################
                # dis, score = line.split()
                if dis in list_name:
                    dis_files_data.append(dis)

        self.data_dict = {'d_img_list': dis_files_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))

        sample = {
            'd_img_org': d_img,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class Koniq10k_resize_crop(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Koniq10k_resize_crop, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []

        #       with open(self.txt_file_name, 'r') as listFile:

        ########################################################### append
        with open(txt_file_name, 'r') as col_names:
            listFile = csv.reader(col_names)
            reader_1 = next(listFile)
            ###########################################################
            for line in listFile:
                #################################append
                dis, score = line[0], line[1]
                #################################
                # dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (448, 448), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]

        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

class Koniq10k_inference_resize_crop(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Koniq10k_inference_resize_crop, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        #       with open(self.txt_file_name, 'r') as listFile:
        ########################################################### append
        with open(txt_file_name, 'r') as col_names:
            listFile = csv.reader(col_names)
            reader_1 = next(listFile)
            ###########################################################
            for line in listFile:
                #################################append
                dis = line[0]
                #################################
                # dis, score = line.split()
                if dis in list_name:
                    dis_files_data.append(dis)

        self.data_dict = {'d_img_list': dis_files_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (448, 448), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))

        sample = {
            'd_img_org': d_img,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class Koniq10k_arg(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Koniq10k_arg, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []

        #       with open(self.txt_file_name, 'r') as listFile:

        ########################################################### append
        with open(txt_file_name, 'r') as col_names:
            listFile = csv.reader(col_names)
            reader_1 = next(listFile)
            ###########################################################
            for line in (listFile):
                #################################append
                dis, score = line[0], line[1]
                #################################
                # dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)
                    for i in range(5):
                        x = i + 1
                        name = str(x)+'_'+dis
                        score = float(score)
                        dis_files_data.append(name)
                        score_data.append(score)


        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]

        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)

        return sample



class Koniq10k_withKADID(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Koniq10k_withKADID, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        # with open(self.txt_file_name, 'r') as listFile:
        ########################################################### append
        with open(txt_file_name, 'r') as col_names:
            listFile = csv.reader(col_names)
            reader_1 = next(listFile)
            ###########################################################
            for line in listFile:
                #################################append
                dis, score = line[0], line[1]
                #################################
                # dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]

        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)

        return sample
