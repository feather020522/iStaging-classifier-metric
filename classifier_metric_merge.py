# thoughts
# 取panorama, room_id, label
# 第一段先分好組, getitem拿pano + room_id
# 第二段分label, getitem拿pano + label
# 依第一段結果分組後, 看誰是primary panorama
# 判斷其他secondary panorama的label是不是大都跟primary一樣
# 完成分組

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse

# import time
import datetime

import random

import torch
from torch.utils.data import Dataset

from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# resnet50 pretrained model import
from torchvision.models import resnet50, ResNet50_Weights

# TensorBoard import
from torch.utils.tensorboard import SummaryWriter

# classification report to print f1 score for each label
from sklearn.metrics import classification_report, precision_recall_curve

from pytorch_metric_learning import distances, losses, miners, reducers, testers

from collections import Counter

from pano_label_getter import *

# PanoLabelGetter()

label_list = ["closet", "bathroom", "bedroom", "living room", "kitchen", "dining room", "garage", "laundry", "hallway", 
"stair", "family room", "breakfast nook", "pantry", "loft", "patio", "office", "doorway", 
"basement", "utility", "balcony", "attic", "yard", "other"]

class CustomZinDBuildingParser(Dataset):
    def __init__(self, buildings, data_dir, transform = None, target_transform = None, mode = "val"):
        self.buildings = buildings
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

    def __len__(self):
        return len(self.buildings)

    def __getitem__(self, idx):
        pano_paths = [os.path.join(self.data_dir, x[0]) for x in self.buildings[idx]]
        # labels = [x[1] for x in self.buildings[idx]]
        labels = [label_list.index(x[1]) for x in self.buildings[idx]]
        room_ids = [x[2] for x in self.buildings[idx]]
        primary_bools = [1 if x[3] == True else 0 for x in self.buildings[idx]]
        id = self.buildings[idx][0][4]

        # if self.mode == "train":
        #     pano_paths, labels = self.GetBalancePosNegPair(pano_paths, labels)

        panos = [Image.open(x) for x in pano_paths]
        panos = [x.resize((256, 128)) for x in panos]
        if self.transform:
            panos = [self.transform(x) for x in panos]
            # for i in range(len(panos)):
            #     panos[i] = np.roll(panos[i], random.randrange(0, 127, 1), axis=2)
            # panos = [np.roll(panos[x], random.randrange(0, 127, 1), axis=2) for x in panos]
        # if self.target_transform:
        #     labels = self.target_transform(labels)

        # convert list to tensor
        panos = [x.tolist() for x in panos]
        panos = torch.tensor(panos)
        # print(panos.size())
        # print(labels)
        labels = torch.tensor(labels)
        room_ids = torch.tensor(room_ids)
        primary_bools = torch.tensor(primary_bools)

        return panos, labels, room_ids, primary_bools, id

class ExtraLayerNet(nn.Module):
    def __init__(self, pretrained_model):
        super(ExtraLayerNet, self).__init__()
        self.pretrained = pretrained_model
        self.new_layers = nn.Linear(1000, 23)
        # self.new_layers = nn.Sequential(nn.Linear(1000, 23),
                                        #    nn.ReLU(),
                                        #    nn.Linear(100, 2))
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.new_layers(x)
        return x
    
class Metric_Net(nn.Module):
    def __init__(self):
        super(Metric_Net, self).__init__()
        # Conv2d: (input_channels, output_channels, kernel_size, stride)
        # 3 * 256 * 128
        self.conv1 = nn.Conv2d(3, 27, 3, 1)
        # 27 * 254 * 126
        self.pool1 = nn.MaxPool2d(2, 2)
        # 27 * 127 * 63
        self.conv2 = nn.Conv2d(27, 81, 9, 3)
        # 81 * 40 * 19
        self.pool2 = nn.MaxPool2d(2, 2)
        # 81 * 20 * 9
        self.conv3 = nn.Conv2d(81, 162, 3, 2)
        # 162 * 9 * 4
        
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(5832, 486)
        self.fc3 = nn.Linear(486, 52)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    


parser = argparse.ArgumentParser(description="Partition Zillow Indoor Dataset (ZInD)")

parser.add_argument(
    "--mode", "-m", help="Trainning mode. T: train, V: val, E: test", nargs='*'
    # nargs='+'
)

parser.add_argument(
    "--model", "-M", help="Input trained model", nargs='*'
    # nargs='+'
)

args = parser.parse_args()
input_file = "./stats.json"
dataset_dir = "/CGVLAB3/datasets/tingwei/zind_data/"
partition_set = "./zind_partition.json"
trainning_mode = args.mode
# trained_model = args.model

mode_train_bool = False
mode_val_bool = False
mode_test_bool = False

if trainning_mode:
    if 'T' in trainning_mode:
        mode_train_bool = True

    if 'V' in trainning_mode:
        mode_val_bool = True

    if 'E' in trainning_mode:
        mode_test_bool = True

with open(input_file, "r") as fh:
    houses_data = json.load(fh)

with open(partition_set, "r") as fh:
    partition_data = json.load(fh)


# pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
# classifier_model = ExtraLayerNet(pretrained_model=pretrained)
classifier_model = resnet50(weights=ResNet50_Weights.DEFAULT)
classifier_model = ExtraLayerNet(pretrained_model = classifier_model)
# classifier_model.fn = nn.Linear(1000, 23)

classifier_model.load_state_dict(torch.load("./classifier_100epochs_resnet50.pth"))
classifier_model.eval()

metric_model = Metric_Net()
metric_model.load_state_dict(torch.load("./metric_model_150.pth"))
metric_model.eval()

# train_panos, train_buildings, train_pano_count, train_grouping_list = PanoLabelGetter(houses_data, partition_data, "train")               # total: 67448 panos, 1575 buildings
# train_panos_val, train_buildings_val, train_pano_count_val = PanoLabelGetter(houses_data, partition_data, "train")               # total: 67448 panos, 1575 buildings
# val_panos, val_buildings, val_pano_count, val_grouping_list = PanoLabelGetter(houses_data, partition_data, "val")
test_panos, test_buildings, test_pano_count, test_grouping_list = PanoLabelGetter(houses_data, partition_data, "test")

transform = transforms.Compose(
    [
        # transforms.ColorJitter(brightness=[0.35, 1.0]),
        transforms.ToTensor(),
    ]
)

room_grouping_test = CustomZinDBuildingParser(
    buildings = test_grouping_list,
    data_dir = dataset_dir,
    transform = transform
)

def my_collate_fn(batch):
    batch = list(batch[0])
    return batch

batch_size = 1
shuffle_train = False
num_workers = 6
# epochs = 150
# val_freq = 15
# train_data = "train partition"
# test_data = "train partition"
# final_val = "val partition"
pos_threshold = 0.75

room_dataloader_test = DataLoader(
    room_grouping_test, 
    batch_size = batch_size, 
    shuffle = shuffle_train, 
    num_workers = num_workers,
    collate_fn = my_collate_fn
)

out_path_metric = "merge_result_id.txt"
# out_path_val = "val_result.txt"
# out_path_status = "training_status.txt"
# fm = open(out_path_metric, 'w')
fm = open(out_path_metric, 'w')
# fma = open("metric_acc.txt", "w")
# fmga = open("metric_group_acc.txt", "w")
# fca = open("classifier_acc.txt", "w")
# fga = open("grouping_acc.txt", "w")
# fv = open(out_path_val, 'w')
# fs = open(out_path_status, 'w')

OOM_count = 16
max_OOM_count = 16

if mode_test_bool:
    print("start testing")
    classifier_model.cuda()
    metric_model.cuda()

    distance = distances.CosineSimilarity()

    for i, data in enumerate(room_dataloader_test):
        panos, labels, room_ids, primary_bools, id = data[0], data[1].tolist(), data[2].tolist(), data[3].tolist(), data[4]

        
        # print("new data")
        # print("new data", file=fm)
        data_len = len(primary_bools)
        # print(data_len, id)
        # print(data_len, file=fm)

        if data_len > max_OOM_count:
            print("continue")
            continue
        print(data_len, id)
        print(data_len, id, file=fm)
        # print()
        room_groups = []
        true_rooms = []
        for idx in range(data_len):
            room_groups.append([idx, -1, -1])
            true_rooms.append([idx, labels[idx], room_ids[idx]])

        # -------- metric part --------
        metric_output = metric_model(panos.cuda())
        print(metric_output)
        print(metric_output.size())
        # break
        mat = distance(metric_output)
        print(mat)
        print(mat.size())

        predicted_ids = []
        correct = 0
        for idx in range(mat.size(dim=1)):
            vec = mat[idx]
            print(vec)
            actual = [1 if x == labels[idx] else 0 for x in labels]
            predicted = [1 if (y - pos_threshold) > 1e-9 else 0 for y in vec]
            print(actual)
            print(predicted)
            # print(predicted, file=fm)
            for z in range(len(actual)):
                correct += actual[z] == predicted[z]
            predicted_ids.append(predicted)
            # break

        # calculate accuracy of metric
        metric_acc = 100.0 * (float)(correct) / (float)(len(predicted_ids) * len(predicted_ids[0]))
        # break

        groups_idx = 0
        for idx, val in enumerate(predicted_ids):
            print(val)
            if room_groups[idx][2] == -1:
                room_groups[idx][2] = groups_idx
                groups_idx += 1
            for j in range(idx+1, len(val)):
                if val[j] == 1:
                    room_groups[j][2] = room_groups[idx][2]

        # calculate accuracy after grouping
        correct = 0
        for i in range(data_len):
            for j in range(data_len):
                if (room_groups[i][2] == room_groups[j][2] and true_rooms[i][2] == true_rooms[j][2]) \
                or (room_groups[i][2] != room_groups[j][2] and true_rooms[i][2] != true_rooms[j][2]):
                    correct += 1

        metric_grouping_acc = 100.0 * (float)(correct) / (float)(data_len * data_len)

        # -------- classifier part --------
        classifier_output = []
        # print(data_len)
        OOM_len = OOM_count
        # print(panos.size())
        while data_len > OOM_len:
            # continue
            # print(OOM_count, OOM_len)
            small_panos = panos[0:OOM_count]
            panos = panos[OOM_count:]
            # print(small_panos.size())
            # print(panos.size())
            OOM_len += OOM_count
            classifier_output.extend(classifier_model(small_panos.cuda()).tolist())
            # print(len(classifier_output))

        # print(panos.size())
        classifier_output.extend(classifier_model(panos.cuda()).tolist())

        cla_acc_list = []
        for idx, lis in enumerate(classifier_output):
            print(idx, lis)
            max_idx, max_val = lis.index(max(lis)), max(lis)
            lis.pop(max_idx)
            val_diff = max_val - max(lis)
            room_groups[idx][1] = [max_idx, val_diff]
            cla_acc_list.append(max_idx)

        # print(cla_acc_list)
        # print(labels)
        correct = 0
        for i in range(data_len):
            correct += cla_acc_list[i] == labels[i]
        classifier_acc = 100.0 * (float)(correct) / (float)(data_len)
        # print(classifier_acc)
        # break


        for i in range(len(room_groups)):
            print(f"{room_groups[i]}, {true_rooms[i]}")
            print(f"{room_groups[i]}, {true_rooms[i]}", file=fm)


        # -------- label grouping --------
        print("Start re-label")
        print("Start re-label", file=fm)


        room_groups = sorted(room_groups, key = lambda x: x[2])

        max_room_id = room_groups[-1][2]
        i, j = 0, 0
        while i <= max_room_id:
            start_j = j
            while j < len(room_groups) and room_groups[j][2] == i:
                j += 1
            # find labels in same group
            # 1st method: calculate appear times
            label_dic = []
            most_possible_label = -1
            for idx in range(start_j, j):
                label_dic.append(room_groups[idx][1][0])
            label_count = dict(Counter(label_dic))
            most_possible_val = max(label_count.values())
            candidate_keys = [key for key, val in label_count.items() if val == most_possible_val]
            if len(candidate_keys) == 1:
                most_possible_label = candidate_keys[0]
            else:
                confidence = -1000
                for idx in range(start_j, j):
                    if room_groups[idx][1][0] in candidate_keys:
                        if confidence < room_groups[idx][1][1]:
                            confidence = room_groups[idx][1][1]
                            most_possible_label = room_groups[idx][1][0]
            
            # 2nd method: sum up all probabilities
            # label_prob = [0] * len(classifier_output[0])
            # for idx in range(start_j, j):
            #     label_prob = [x + y for x, y in zip(label_prob, classifier_output[room_groups[idx][0]])]
            #     print(label_prob)
            # most_possible_label = label_prob.index(max(label_prob))


            # most_possible_label = max(label_dic, key = label_dic.count)
            for idx in range(start_j, j):
                room_groups[idx][1] = most_possible_label
            
            i += 1

        room_groups = sorted(room_groups, key = lambda x: x[0])
        correct = 0
        for i in range(data_len):
            correct += room_groups[i][1] == labels[i]
            print(f"{room_groups[i]}, {true_rooms[i]}")
            print(f"{room_groups[i]}, {true_rooms[i]}", file=fm)
        label_group_acc = 100.0 * (float)(correct) / (float)(data_len)
        # print(group_acc)    
        
        metric_acc_str = f"Metric accuracy: {metric_acc} %"
        metric_group_acc_str = f"Grouping accuracy: {metric_grouping_acc} %"
        cla_acc_str = f"Classifier accuracy: {classifier_acc} %"
        cla_group_acc_str = f"Label accuracy after grouping: {label_group_acc} %"
        print(metric_acc_str)
        print(metric_acc_str, file=fm)
        print(metric_acc_str, file=fma)
        print(metric_group_acc_str)
        print(metric_group_acc_str, file=fm)
        print(metric_group_acc_str, file=fmga)
        print(cla_acc_str)
        print(cla_acc_str, file=fm)
        print(cla_acc_str, file=fca)
        print(cla_group_acc_str)
        print(cla_group_acc_str, file=fm)
        print(cla_group_acc_str, file=fga)
        # break
        
        # print(room_ids.tolist())
        # print(room_ids.tolist(), file=fm)

    print("finish testing")
