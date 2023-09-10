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

# TensorBoard import
from torch.utils.tensorboard import SummaryWriter

# classification report to print f1 score for each label
from sklearn.metrics import classification_report, precision_recall_curve

from collections import Counter

# dataloader for building

filter_size = 48

class CustomZinDBuildingParser(Dataset):
    def __init__(self, buildings, data_dir, transform, mode = "val"):
        self.buildings = buildings
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.buildings)

    def __getitem__(self, idx):
        pano_paths = [os.path.join(self.data_dir, x[0]) for x in self.buildings[idx]]
        labels = [x[1] for x in self.buildings[idx]]

        if self.mode == "train":
            pano_paths, labels = self.GetBalancePosNegPair(pano_paths, labels)

        panos = [Image.open(x) for x in pano_paths]
        panos = [x.resize((256, 128)) for x in panos]
        if self.transform:
            panos = [self.transform(x) for x in panos]
            for i in range(len(panos)):
                panos[i] = np.roll(panos[i], random.randrange(0, 127, 1), axis=2)
            # panos = [np.roll(panos[x], random.randrange(0, 127, 1), axis=2) for x in panos]

        # convert list to tensor
        panos = [x.tolist() for x in panos]
        panos = torch.tensor(panos)
        # print(panos.size())
        labels = torch.tensor(labels)

        return panos, labels
    
    def GetBalancePosNegPair(self, ip_panos, ip_labels):
        panos = []
        labels = []
        # count = dict(Counter(labels))
        # print(ip_labels)
        max_label = max(ip_labels)
        label_cnt = min(ip_labels)

        # random get 2
        while (label_cnt < max_label):
            # if (count.get(label_cnt) == None):
            cur_label_keys = [k for k, v in enumerate(ip_labels) if v == label_cnt]
            # print(cur_label_keys)
            cur_get_labels = random.sample(cur_label_keys, 2)
            for j in cur_get_labels:
                panos.append(ip_panos[j])
                labels.append(ip_labels[j])
            label_cnt += 1

        # Get the first 2
        # for idx, label in enumerate(ip_labels):
        #     c = 0 
        #     if count.get(label):
        #         c = count[label]
        #     # print(c)
        #     if c < 2:
        #         panos.append(ip_panos[idx])
        #         labels.append(ip_labels[idx])
        #     count = dict(Counter(labels))
        
        return panos, labels
            

def PanoLabelGetter(house_data, partition_data, train_type, need_to_cut = False):
    # out_path_stat = "pano_count" + train_type + ".txt"
    # fd = open(out_path_stat, 'w')
    label_pairs = []            # store img paths
    building_room_pairs = []
    original_pano_count_list = []
    room_counter = {}
    # j = 0 
    # print(train_type, file=fd)
    print(train_type)
    for id, houses in house_data.items():
            if id in partition_data[train_type]:
                i = 0
                room_pairs = []
                for floor_id, floor_data in houses.items():
                    room_label_a = floor_id.split("_")[1]
                    for complete_room_data in floor_data.values():
                        for partial_room_id, partial_room_data in complete_room_data.items():
                            room_label_b = partial_room_id.split("_")[2]
                            for pano_id, pano_data in partial_room_data.items():
                                room_label = room_label_a + room_label_b
                                pano_path = os.path.join(id, pano_data["image_path"])
                                label = pano_data["new_label"]
                                label_pairs.append([pano_path, label])
                                room_pairs.append([pano_path, room_label])
                                i += 1

                # if filter_size == 0:
                #     continue
                # fix label here
                # transform label 4-code to 0~n   
                j = 0
                new_labels = []
                transdict = {}
                for room in room_pairs:
                    if room[1] not in transdict:
                        transdict[room[1]] = j
                        j += 1
                    new_labels.append(transdict[room[1]])
                    room[1] = transdict[room[1]]
                
                # print(f"ID: {id}, pano數: {i}, label數(有幾間partial room): {j}")

                # def PanoCutter():

                room_pairs_2 = []
                new_labels_2 = []
                if i > 72:
                    # pano count > 72 => cut to half
                    room_pairs_2 += room_pairs[i//2:]
                    new_labels_2 += new_labels[i//2:]
                    room_pairs = room_pairs[:i//2]
                    new_labels = new_labels[:i//2]

                    # Expand panos
                    room_pairs_2, new_labels_2 = PanoExpander(room_pairs_2, new_labels_2)
                    room_pairs_2 = sorted(room_pairs_2, key = lambda x: x[1])
                    

                    # need to let pos / neg pair balance
                    # cut label count -> 1 ~ 11
                    if need_to_cut:
                        min_label = min(room_pairs_2, key = lambda x: x[1])[1]
                        max_label = max(room_pairs_2, key = lambda x: x[1])[1]
                        batch_count = 6
                        while max_label - min_label + 1 >= 2 * batch_count:
                            new_room_pair = [k for k in room_pairs_2 if k[1] >= min_label and k[1] < min_label + batch_count]
                            room_pairs_2 = [k for k in room_pairs_2 if k[1] >= min_label + batch_count]
                            min_label = min(room_pairs_2, key = lambda x: x[1])[1]
                            building_room_pairs.append(new_room_pair)

                    building_room_pairs.append(room_pairs_2)

                room_pairs, new_labels = PanoExpander(room_pairs, new_labels)
                room_pairs = sorted(room_pairs, key = lambda x: x[1])
                if need_to_cut:
                    min_label = min(room_pairs, key = lambda x: x[1])[1]
                    max_label = max(room_pairs, key = lambda x: x[1])[1]
                    batch_count = 6
                    while max_label - min_label + 1 >= 2 * batch_count:
                        new_room_pair = [k for k in room_pairs if k[1] >= min_label and k[1] < min_label + batch_count]
                        room_pairs = [k for k in room_pairs if k[1] >= min_label + batch_count]
                        min_label = min(room_pairs, key = lambda x: x[1])[1]
                        building_room_pairs.append(new_room_pair)

                # pos_pair = 0
                # neg_pair = 0
                # for iidx, ii in enumerate(new_labels):
                #     for jjdx, jj in enumerate(new_labels):
                #         if iidx != jjdx:
                #             if ii == jj:
                #                 pos_pair += 1
                #             else:
                #                 neg_pair += 1
                # print(f"{id}, {pos_pair//2}, {neg_pair//2}, {neg_pair / pos_pair}")
                # print(neg_pair / pos_pair)
                # if neg_pair / pos_pair < 5 or neg_pair / pos_pair > 80:
                    # print(new_labels)

                # print(f"label len: {len(new_labels)}")

                # if (len(new_labels) > 80):
                #     print("So long")

                building_room_pairs.append(room_pairs)

    # print(room_counter)
    return label_pairs, building_room_pairs, original_pano_count_list

def PanoExpander(pano_list: list, label_list: list):
    # get each label count
    count = dict(Counter(label_list))
    # expand pano which count < 2 (no positive pair)
    while(min(count.values()) <= 1):
        min_keys = [k for k, v, in count.items() if v == min(count.values())]
        roll_label = random.choice(min_keys)            # random get min label
        roll_index = label_list.index(roll_label)       # get label's index
        pano_list.append(pano_list[roll_index])         # duplicate pano
        label_list.append(roll_label)                   # duplicate label
        count = dict(Counter(label_list))
    return pano_list, label_list


# building_room_pairs sturcture:
#        building_list
#           /       \
#   one building  one building
#       /
# pano, label pairs

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
trained_model = args.model

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

print("\nnew data")

train_panos, train_buildings, train_pano_count = PanoLabelGetter(houses_data, partition_data, "train", need_to_cut = True)               # total: 67448 panos, 1575 buildings
train_panos_val, train_buildings_val, train_pano_count_val = PanoLabelGetter(houses_data, partition_data, "train")               # total: 67448 panos, 1575 buildings
val_panos, val_buildings, val_pano_count = PanoLabelGetter(houses_data, partition_data, "val")
test_panos, test_buildings, test_pano_count = PanoLabelGetter(houses_data, partition_data, "test")
# test_panos_val, test_buildings_val, test_pano_count_val = PanoLabelGetter(houses_data, partition_data, "test")

# exit()

transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=[0.35, 1.0]),
        transforms.ToTensor(),
    ]
)

room_identifier_train = CustomZinDBuildingParser(
    buildings = train_buildings,
    data_dir = dataset_dir,
    transform = transform,
    mode = "train"
)

room_identifier_train_val = CustomZinDBuildingParser(
    buildings = train_buildings_val,
    data_dir = dataset_dir,
    transform = transform
)

room_identifier_val = CustomZinDBuildingParser(
    buildings = val_buildings,
    data_dir = dataset_dir,
    transform = transform
)

room_identifier_test = CustomZinDBuildingParser(
    buildings = test_buildings,
    data_dir = dataset_dir,
    transform = transform
)

# room_identifier_test_val = CustomZinDBuildingParser(
#     buildings = test_buildings_val,
#     data_dir = dataset_dir,
#     transform = transform
# )

batch_size = 1
shuffle_train = False
num_workers = 6
epochs = 200
val_freq = 20
train_data = "train partition"
test_data = "train partition"
final_val = "val partition"
pos_threshold = 0.75
def my_collate_fn(batch):
    batch = list(batch[0])
    return batch

room_dataloader_train = DataLoader(
    room_identifier_train, 
    batch_size = batch_size, 
    shuffle = shuffle_train, 
    num_workers = num_workers,
    collate_fn = my_collate_fn
)

room_dataloader_train_val = DataLoader(
    room_identifier_train_val, 
    batch_size = batch_size, 
    shuffle = shuffle_train, 
    num_workers = num_workers,
    collate_fn = my_collate_fn
)

room_dataloader_val = DataLoader(
    room_identifier_val, 
    batch_size = batch_size, 
    shuffle = False, 
    num_workers = num_workers,
    collate_fn = my_collate_fn
)

room_dataloader_test = DataLoader(
    room_identifier_test, 
    batch_size = batch_size, 
    shuffle = shuffle_train, 
    num_workers = num_workers,
    collate_fn = my_collate_fn
)

# room_dataloader_test_val = DataLoader(
#     room_identifier_test_val, 
#     batch_size = batch_size, 
#     shuffle = shuffle_train, 
#     num_workers = num_workers,
#     collate_fn = my_collate_fn
# )

# exit()

# -------- neural network --------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
    
net = Net()

losses_list = []
correct_list = []

if mode_train_bool:

    net.cuda()
    # ts_writer = SummaryWriter(log_dir = "./test")
    ts_writer = SummaryWriter()

    import torch.optim as optim
    import faiss

    from pytorch_metric_learning import distances, losses, miners, reducers, testers
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # output file
    out_path_train = "train_result.txt"
    out_path_val = "val_result.txt"
    out_path_status = "training_status.txt"
    ft = open(out_path_train, 'w')
    fv = open(out_path_val, 'w')
    fs = open(out_path_status, 'w')

    nowTime = datetime.datetime.now() # 取得現在時間
    
    output_str = f"parameters: \nbatch size = {batch_size}\nshuffle train = {shuffle_train}\nnum workers = {num_workers}\nepoch = {epochs}\ntest set = {test_data}\npos threshold = {pos_threshold}"
    print(output_str)
    print(output_str, file=fs)

    output_str = f"Start training, {nowTime}"
    print(output_str)
    print(output_str, file=fs)

    loss_freq = 50

    def my_acc_calculator(val_dataloader, model, distance, epoch = 0, val_dataset = "test"):
                
        actual_label = []
        predicted_label = []
        predicted_label_PR = []
        target_label = ['negative pair', 'positive pair']
        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            mat = distance(outputs)
            # print(mat.device)
            # print(mat.size(dim=1))
            # same_dises = []
            for idx in range(mat.size(dim=1)):
                vec = mat[idx]
                actual = [1 if x == labels[idx] else 0 for x in labels]
                predicted = [1 if (y - pos_threshold) > 1e-9 else 0 for y in vec]
                
                actual_label += actual
                predicted_label += predicted
                predicted_label_PR += vec.tolist()

        total = len(actual_label)
        # print(actual_label)
        # print(predicted_label)
        correct = 0
        for i in range(total):
            correct += actual_label[i] == predicted_label[i]
        accuracy = 100.0 * (float)(correct) / (float)(total) 
        output_str = f"Test set accuracy (pos/neg pair) at epoch {epoch + 1} on dataset {val_dataset} = {accuracy}"
        print(output_str, file=fv)
        ts_writer.add_scalar(f"accuracy per {val_freq} epoch on dataset {val_dataset}", accuracy, epoch + 1)


        report = classification_report(actual_label, predicted_label, target_names=target_label)
        print(report)
        print(report, file=fs)
        # precision, recall, thresholds = precision_recall_curve(actual_label, predicted_label_PR)
        # if epoch == epochs - 1:
        ts_writer.add_pr_curve(f"PR curve per {val_freq} epoch on dataset {val_dataset}", np.array(actual_label), np.array(predicted_label_PR))
        # report = classification_report(actual_label, predicted_label)
        return report

# def train(dataset, epochs):
    # training
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(room_dataloader_train, 0):

            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            # print(outputs.size())
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # loss.item()
            if i % loss_freq == loss_freq - 1 or i == len(room_dataloader_train) - 1:
                running_loss /= (i % loss_freq + 1)
                output_str = f'[{epoch + 1}] [data:{i+1}] loss: {running_loss:.6f}'
                print(output_str, file=ft)
                losses_list.append(running_loss)
                ts_writer.add_scalar(f"Loss per {loss_freq} mini_batch", running_loss, epoch * len(room_dataloader_train) + (i + 1))
                running_loss = 0.0

            # print("pass")

        # get current time
        nowTime = datetime.datetime.now()
        output_str = f"epoch # {epoch + 1} done, {nowTime}"
        print(output_str)
        print(output_str, file=fs)

        # validation            

        if epoch % val_freq == val_freq - 1 and mode_val_bool:
            # acc_report = my_acc_calculator(room_dataloader_test_val, net, distance, epoch)            
            my_acc_calculator(room_dataloader_test, net, distance, epoch)            
            # print val time
            nowTime = datetime.datetime.now()
            output_str = f"val # {epoch + 1} on test done, {nowTime}"
            print(output_str)
            print(output_str, file=fs)

            # on train
            my_acc_calculator(room_dataloader_train_val, net, distance, epoch)            
            # print val time
            nowTime = datetime.datetime.now()
            output_str = f"val # {epoch + 1} on train done, {nowTime}"
            print(output_str)
            print(output_str, file=fs)
            # print(acc_report)
            # print(acc_report, file=fs)
            
        if epoch == epochs - 1:
            acc_report = my_acc_calculator(room_dataloader_val, net, distance, epoch)
            # last val: val
            nowTime = datetime.datetime.now()
            output_str = f"Last val: val done, {nowTime}"
            # print(acc_report)
            # print(acc_report, file=fs)
            print(output_str)
            print(output_str, file=fs)

    # train(room_dataloader_test, epochs)
    ts_writer.flush()
    ts_writer.close()
    print('Finished Training')
    print('Finished Training', file=fs)

# def my_acc_calculator(val_dataloader, model, distance):
                
#                 actual_label = []
#                 predicted_label = []
#                 target_label = ['negative pair', 'positive pair']
#                 for i, data in enumerate(val_dataloader, 0):
#                     inputs, labels = data[0].cuda(), data[1].cuda()
#                     outputs = model(inputs)
#                     mat = distance(outputs)
#                     # print(mat)
#                     # print(mat.size(dim=1))
#                     # same_dises = []
#                     for idx in range(mat.size(dim=1)):
#                         vec = mat[idx]
#                         # print(vec)
#                         # print(labels)
#                         actual = [1 if x == labels[idx] else 0 for x in labels]
#                         predicted = [1 if (y - 0.85) > 1e-9 else 0 for y in vec]
                        
#                         actual_label += actual
#                         predicted_label += predicted
#                         # check each distance >= specific value -> true, else false
#                         # value = 0.85

#                         # same_dis = [(x2, x.item()) for x2, x in enumerate(vec) if labels[idx] == labels[x2]]
#                         # same_dis2 = [1 if x[1] >= 0.85 else 0 for x in same_dis]
#                         # print(same_dis2)
#                         # same_dises.append(same_dis2)

#                         # find nearest neighbor
                        
#                         # vec[idx] = 0.0
#                         # max_idx = torch.where(vec == torch.max(vec))[0].tolist()[0]
#                         # print(max_idx)
#                         # print(f"index this: {idx}, nearest: {max_idx}")
#                         # print(f"distance this: {labels[idx].item()}, nearest: {labels[max_idx].item()}")
#                 total = len(actual_label)
#                 correct = (actual_label == predicted_label).sum().item()
#                 accuracy = 100.0 * (float)(correct) / (float)(total) 
#                 output_str = f"Test set accuracy (pos/neg pair) at epoch {epoch + 1} = {accuracy}"
#                 print(output_str, file=fv)
#                 ts_writer.add_scalar(f"accuracy per {val_freq} epoch", accuracy, epoch + 1)


#                 report = classification_report(actual_label, predicted_label, target_names=target_label)
#                 # report = classification_report(actual_label, predicted_label)
#                 return report