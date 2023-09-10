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
from sklearn.metrics import classification_report

from collections import Counter

# dataloader for building

filter_size = 48

class CustomZinDBuildingParser(Dataset):
    def __init__(self, buildings, data_dir, transform, pano_count=[], filter=False):
        self.buildings = buildings
        self.data_dir = data_dir
        self.transform = transform
        self.pano_count = pano_count
        self.filter = filter

    def __len__(self):
        return len(self.buildings)

    def __getitem__(self, idx):
        pano_paths = [os.path.join(self.data_dir, x[0]) for x in self.buildings[idx]]
        labels = [x[1] for x in self.buildings[idx]]

        panos = [Image.open(x) for x in pano_paths]
        panos = [x.resize((256, 128)) for x in panos]
        if self.transform:
            panos = [self.transform(x) for x in panos]

        # transform label 4-code to 0~n
        # j = 0
        # new_labels = []
        # transdict = {}
        # for i in labels:
        #     if i not in transdict:
        #         transdict[i] = j
        #         j += 1
        #     new_labels.append(transdict[i])
        
        # if self.filter:
        #     while len(new_labels) < filter_size:
        #         count = dict(Counter(new_labels))
        #         min_keys = [k for k, v, in count.items() if v == min(count.values())]
        #         roll_label = random.choice(min_keys)
        #         roll_index = new_labels.index(roll_label)
        #         extra_pano = np.roll(panos[roll_index], random.randrange(0, 127, 1), axis=2)
        #         panos.append(extra_pano)
        #         new_labels.append(roll_label)        
        
        if self.pano_count:
            j = self.pano_count[idx]
            while j < len(panos):
                panos[j] = np.roll(panos[j], random.randrange(0, 127, 1), axis=2)
                j += 1


        # convert list to tensor
        panos = [x.tolist() for x in panos]
        panos = torch.tensor(panos)
        # print(panos.size())
        labels = torch.tensor(labels)
        # print(new_labels)
        # temp = []
        # for j, x in enumerate(panos):
        #     temp += x.tolist()
        # temp2 = []
        # for j, x in enumerate(new_labels):
        #     temp2.append(int(x[0]))
        # temp = torch.tensor(temp)
        # temp2 = torch.tensor(temp2)

        return panos, labels
            

def PanoLabelGetter(house_data, partition_data, train_type, filter_size=0, expandlist=False):
    # out_path_stat = "pano_count" + train_type + ".txt"
    # fd = open(out_path_stat, 'w')
    label_pairs = []            # store img paths
    building_room_pairs = []
    original_pano_count_list = []
    # j = 0 
    # print(train_type, file=fd)
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

                if filter_size and i > filter_size:
                    continue
                # fix label here
                j = 0
                new_labels = []
                transdict = {}
                for room in room_pairs:
                    if room[1] not in transdict:
                        transdict[room[1]] = j
                        j += 1
                    room[1] = transdict[room[1]]
                    new_labels.append(j)

                # expand batch size to filter
                original_pano_count_list.append(i)
                if expandlist:
                    while len(new_labels) < filter_size:
                        count = dict(Counter(new_labels))
                        min_keys = [k for k, v, in count.items() if v == min(count.values())]
                        roll_label = random.choice(min_keys)
                        roll_index = new_labels.index(roll_label)
                        # extra_pano = np.roll(panos[roll_index], random.randrange(0, 127, 1), axis=2)
                        # panos.append(extra_pano)
                        # extra_room_pair = room_pairs[roll_index]
                        room_pairs.append(room_pairs[roll_index])
                        new_labels.append(roll_label)
                # print(len(room_pairs))
                building_room_pairs.append(room_pairs)

    return label_pairs, building_room_pairs, original_pano_count_list

          

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

train_panos, train_buildings, train_pano_count = PanoLabelGetter(houses_data, partition_data, "train")               # total: 67448 panos, 1575 buildings
val_panos, val_buildings, val_pano_count = PanoLabelGetter(houses_data, partition_data, "val", filter_size, expandlist = True)
test_panos, test_buildings, test_pano_count = PanoLabelGetter(houses_data, partition_data, "test", filter_size, expandlist = True)

# MetricTestFilter(test_buildings, 48)
# for i in test_buildings:
#     print(len(i))

# exit()

transform = transforms.Compose(
    [
        # transforms.ColorJitter(brightness=[0.35, 1.0]),
        transforms.ToTensor(),
    ]
)

# room_identifier_train = CustomZinDBuildingParser(
#     buildings = train_buildings,
#     data_dir = dataset_dir,
#     transform = transform
# )

room_identifier_val = CustomZinDBuildingParser(
    buildings = val_buildings,
    data_dir = dataset_dir,
    transform = transform,
    pano_count = val_pano_count,
    filter = True
)

room_identifier_test = CustomZinDBuildingParser(
    buildings = test_buildings,
    data_dir = dataset_dir,
    transform = transform,
    pano_count = test_pano_count,
    filter = True
)

batch_size = 1
shuffle_train = False
num_workers = 6
epochs = 20
def my_collate_fn(batch):
    batch = list(batch[0])
    return batch

# room_dataloader_train = DataLoader(
#     room_identifier_train, 
#     batch_size = batch_size, 
#     shuffle = shuffle_train, 
#     num_workers = num_workers
# )

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

# for i, data in enumerate(room_dataloader_val, 0):
#     print(data)
#     break
    # inputs, labels = data[0], data[1]

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
        
        # current OK, adjust model later
        # self.conv3 = nn.Conv2d(81, )
        
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(14580, 660)
        self.fc3 = nn.Linear(660, 132)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
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
    ts_writer = SummaryWriter()

    import torch.optim as optim
    import faiss

    from pytorch_metric_learning import distances, losses, miners, reducers, testers
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # output file
    out_path_train = "train_result.txt"
    out_path_val = "val_result.txt"
    out_path_status = "training_status.txt"
    ft = open(out_path_train, 'w')
    fv = open(out_path_val, 'w')
    fs = open(out_path_status, 'w')

    nowTime = datetime.datetime.now() # 取得現在時間
    
    output_str = f"parameters: \nbatch size = {batch_size}\nshuffle train = {shuffle_train}\nnum workers = {num_workers}\nepoch = {epochs}\nmini batch size = {filter_size}"
    print(output_str)
    print(output_str, file=fs)

    output_str = f"Start training, {nowTime}"
    print(output_str)
    print(output_str, file=fs)

    loss_freq = 50

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(room_dataloader_test, 0):

            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs.size())
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # loss.item()
            if i % loss_freq == loss_freq - 1 or i == len(room_dataloader_test) - 1:
                running_loss /= (i % loss_freq + 1)
                output_str = f'[{epoch + 1}] [data:{i+1}] loss: {running_loss:.6f}'
                print(output_str, file=ft)
                losses_list.append(running_loss)
                ts_writer.add_scalar(f"Loss per {loss_freq} mini_batch", running_loss, epoch * len(room_dataloader_test) + (i + 1))
                running_loss = 0.0

            # print("pass")

        # get current time
        nowTime = datetime.datetime.now()
        output_str = f"epoch # {epoch + 1} done, {nowTime}"
        print(output_str)
        print(output_str, file=fs)

        # validation
        if epoch % 1 == 0 and mode_val_bool:
            correct = 0
            total = 0
            final_arr_predict = []
            final_arr_real = []
            ### convenient function from pytorch-metric-learning ###
            def get_all_embeddings(dataset, model):
                tester = testers.BaseTester(batch_size = batch_size)
                return tester.get_all_embeddings(dataset, model, collate_fn = my_collate_fn)


            ### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
            def test(train_set, test_set, model, accuracy_calculator):
                train_embeddings, train_labels = get_all_embeddings(train_set, model)
                test_embeddings, test_labels = get_all_embeddings(test_set, model)
                train_labels = train_labels.squeeze(1)
                test_labels = test_labels.squeeze(1)
                print("Computing accuracy")
                accuracies = accuracy_calculator.get_accuracy(
                    test_embeddings, test_labels, train_embeddings, train_labels, False
                )
                output_str = f"Test set accuracy (Precision@1) at epoch {epoch + 1} = {accuracies['precision_at_1']}"
                print(output_str, file=fv)
                correct_list.append(accuracies['precision_at_1'])
                ts_writer.add_scalar("accuracy per 2 epoch", accuracies['precision_at_1'], epoch + 1)

            test(room_identifier_test, room_identifier_val, net, accuracy_calculator)

            # with torch.no_grad():
            #     for data in room_dataloader_val:
            #         # data to tensor
            #         temp = []
            #         for j, x in enumerate(data[0]):
            #             temp += x.tolist()
            #         temp2 = []
            #         for j, x in enumerate(data[1]):
            #             temp2.append(int(x[0]))
            #         temp = torch.tensor(temp)
            #         temp2 = torch.tensor(temp2)
            #         images, labels = temp.cuda(), temp2.cuda()
                    
            #         # calculate outputs by running images through the network
            #         outputs = net(images)
            #         # the class with the highest energy is what we choose as prediction
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels).sum().item()
            #         if epoch == epochs - 1:
            #             final_arr_predict += predicted
            #             final_arr_real += labels
            # accuracy = 100.0 * (float)(correct) / (float)(total)
            # output_str = f'Accuracy of the network on the epoch no. {epoch + 1}: {accuracy} %'
            # print(output_str, file=fv)
            # correct_list.append(accuracy)
            # ts_writer.add_scalar("accuracy per 2 epoch", accuracy, epoch + 1)
            
            # print val time
            nowTime = datetime.datetime.now()
            output_str = f"val # {epoch + 1} done, {nowTime}"
            print(output_str)
            print(output_str, file=fs)
        
            # # save & val model
            # if epoch == epochs - 1:
            #     if trained_model:
            #         model_path = f'./zillow_net_{epoch + 51}epochs_resnet50.pth'
            #     else:
            #         model_path = f'./zillow_net_{epoch + 1}epochs_resnet50.pth'
            #     # torch.save(net.state_dict(), model_path)
            #     final_arr_predict = torch.tensor(final_arr_predict, device='cpu')
            #     final_arr_real = torch.tensor(final_arr_real, device='cpu')
            #     report = classification_report(final_arr_real, final_arr_predict)
            #     print(report)
            #     print(report, file=fs)


    ts_writer.flush()
    ts_writer.close()
    print('Finished Training')
    print('Finished Training', file=fs)
