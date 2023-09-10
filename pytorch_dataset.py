import os
import json
import argparse
import random

import time
import datetime

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
# from torchvision.io import read_image
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms

# labels

label_list = ["closet", "bathroom", "bedroom", "living room", "kitchen", "dining room", "garage", "laundry", "hallway", 
"stair", "family room", "breakfast nook", "pantry", "loft", "patio", "office", "doorway", 
"basement", "utility", "balcony", "attic", "yard", "other"]

class CustomZinDLabelParser(Dataset):
    def __init__(self, pano_pairs, data_dir, transform=None, target_transform=None, train=False):
        self.pano_pairs = pano_pairs                            # dictionary under house
        self.data_dir = data_dir                                # "data"
        self.transform = transform
        self.target_transform = target_transform
        self.train = train                                      # to determine current dataset is Train or not
        
    def __len__(self):
        return len(self.pano_pairs)
    
    def __getitem__(self, idx):
        pano_path = os.path.join(self.data_dir, self.pano_pairs[idx][0])

        pano = Image.open(pano_path)
        label = label_list.index(self.pano_pairs[idx][1])
        pano = pano.resize((512, 256))
        if self.transform:
            pano = self.transform(pano)
        if self.target_transform:
            label = self.target_transform(label)
        # pano = pano.swapaxes(0, 1)
        # pano = pano.swapaxes(1, 2)
        # pano = pano.to(torch.float)
        # pano = torch.div(pano, 255)
        # pano = fn.normalize(pano, mean=[0.5000], std=[.1000])              # fix here
        return pano, label
                            
def PanoLabelGetter(house_data, partition_data, type):
    pano_pairs = []                                    # store img paths
    for id, houses in house_data.items():
            if id in partition_data[type]: 
                for floor_id, floor_data in houses.items():
                    for complete_room_data in floor_data.values():
                        for partial_room_id, partial_room_data in complete_room_data.items():
                            for pano_id, pano_data in partial_room_data.items():
                                pano_path = os.path.join(id, pano_data["image_path"])
                                label = pano_data["new_label"]
                                pano_pairs.append([pano_path, label])
                                # self.pano_paths.append(os.path.join(id, pano_data["image_path"]))
                                # self.labels.append(pano_data["new_label"])
    return pano_pairs

parser = argparse.ArgumentParser(description="Partition Zillow Indoor Dataset (ZInD)")

parser.add_argument(
    "--input", "-i", help="Input json file contains all the home tours.", required=True
    # nargs='+'
)

parser.add_argument(
    "--partition", "-p", help="Input data partition."
)

parser.add_argument(
    "--mode", "-m", help="Trainning mode. T: train, V: val, E: test", nargs='+'
)


# parser.add_argument(
#     "--output", "-o", help="Output folder where zind_partition.json will be saved to", required=True
# )

args = parser.parse_args()
input_file = args.input
partition_set = args.partition
trainning_mode = args.mode

mode_train_bool = False
mode_val_bool = False

# print(trainning_mode)

if 'T' in trainning_mode:
    mode_train_bool = True

if 'V' in trainning_mode:
    mode_val_bool = True

with open(input_file, "r") as fh:
    houses_data = json.load(fh)

with open(partition_set, "r") as fh:
    partition_data = json.load(fh)

# print(partition_data)

# pano_pairs = PanoLabelGetter(houses_data)               # total: 67448 panos
train_panos = PanoLabelGetter(houses_data, partition_data, "train")               # total: 67448 panos
val_panos = PanoLabelGetter(houses_data, partition_data, "val")
test_panos = PanoLabelGetter(houses_data, partition_data, "test")

# -------- old data partitioner --------

# train_ratio = 0.8
# val_ratio = 0.1
# test_ratio = 0.1
# pano_size = len(pano_pairs)

# # Part 1: use CustomZinDLabelParser to identify room label

# num_train = int(pano_size * train_ratio)
# num_val = int(pano_size * val_ratio)

# random.shuffle(pano_pairs)

# train_panos = pano_pairs[:num_train]
# val_panos = pano_pairs[num_train:num_train + num_val]
# test_panos = pano_pairs[num_train + num_val:]

# -------- Dataloader --------

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


label_identifier_train = CustomZinDLabelParser(
    pano_pairs = train_panos,
    data_dir = "data",
    transform = transforms.ToTensor(),
    train = True
)

label_identifier_val = CustomZinDLabelParser(
    pano_pairs = val_panos,
    data_dir = "data",
    transform = transforms.ToTensor()
)

label_identifier_test = CustomZinDLabelParser(
    pano_pairs = test_panos,
    data_dir = "data"
)

batch_size = 128
label_dataloader_train = DataLoader(label_identifier_train, batch_size = batch_size, shuffle = True, num_workers = 2)
label_dataloader_val = DataLoader(label_identifier_val, batch_size = batch_size, shuffle = False)

# -------- neural network --------

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # image shape: 3 * 256 * 512, color 3 channels
        # Conv2d: (input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # current shape: 6 * 252 * 508
        # Maxpool: (kernel_size, stride)
        # take max in kernel * kernel, move stride 
        self.pool = nn.MaxPool2d(2, 2)
        # current shape: 6 * 126 * 254
        self.conv2 = nn.Conv2d(6, 16, 5)
        # current shape: 16 * 122 * 250 
        # max pooling twice -> 16 * 61 * 125
        self.fc1 = nn.Linear(16 * 61 * 125, 240)
        self.fc2 = nn.Linear(240, 84)
        self.fc3 = nn.Linear(84, 23)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

losses = []
correct_list = []

if mode_train_bool:

    net.to(device)          # train on GPU

    # -------- start to train --------

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # output file
    out_path_train = "train_result.txt"
    out_path_val = "val_result.txt"
    ft = open(out_path_train, 'w')
    fv = open(out_path_val, 'w')

    nowTime = datetime.datetime.now() # 取得現在時間
    print(f"start Trainning, {nowTime}")

    for epoch in range(50):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(label_dataloader_train, 0):
        
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # loss.item()
            if i % 50 == 49 or i == len(label_dataloader_train) - 1:
                output_str = f'[{epoch + 1}] [data:{i+1}] loss: {running_loss / (i % 50 + 1):.6f}'
                print(output_str, file=ft)
                losses.append(running_loss / (i % 50 + 1))
                running_loss = 0.0

        # get current time
        nowTime = datetime.datetime.now()
        print(f"epoch # {epoch + 1} done, {nowTime}")

        # validation
        if epoch % 5 == 4 and mode_val_bool:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in label_dataloader_val:
                    images, labels = data[0].to(device), data[1].to(device)
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100.0 * (float)(correct) / (float)(total)
            output_str = f'Accuracy of the network on the epoch no. {epoch}: {accuracy} %'
            print(output_str, file=fv)
            correct_list.append(accuracy)
            # # save model
            # model_path = './zillow_net_' + (str)(epoch) + '.pth'
            # torch.save(net.state_dict(), model_path)


    print('Finished Training')


# -------- print loss curve --------

# use train_result.txt to test draw

# epoch_loss = []
# epoch_size = 845
loss_temp = 0.0
if not mode_train_bool: 
    print("print only")
    train_result = open('train_result.txt')
    val_result = open('val_result.txt')
    i = 0
    for n, line in enumerate(train_result):
        # if "[1]" in line:
        # losses.append((float)(line.split(" ")[3].split("\n")[0]))
        loss_temp = ((float)(line.split(" ")[3].split("\n")[0]))
        loss_temp *= 50
        loss_temp /= (float)(((int)(line.split(" ")[1].split(":")[1].split("]")[0]) - 1) % 50 + 1) 
        losses.append(loss_temp)
        # i += 1
        
        # if i % 50 == 0:
        #     loss_temp /= 50.0
        #     losses.append(loss_temp)
        #     loss_temp = 0.0
        # elif i % epoch_size == 0:
        #     loss_temp /= 45.0
        #     losses.append(loss_temp)
        #     loss_temp = 0.0
        #     i = 0 
    
    i = 0
    for n, line in enumerate(val_result):
        correct_list.append((float)(line.split(" ")[9]))

# print(losses)

x0 = np.arange(0, 50, 50.0 / len(losses))
# print(x0)

# x1 = range(len(losses))
y1 = losses
print(len(losses))
plt.figure()
plt.plot(x0, y1)
plt.title('loss of each epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()
plt.savefig("loss_pic_test.jpg")

x2 = range(0, 50, 5)
y2 = correct_list
plt.figure()
plt.plot(x2, y2)
plt.title('accuracy per 5 epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(0, 100)
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
# plt.show()
plt.savefig("accuracy_pic_test.jpg")
# plt.ylim(40, 60)
# plt.savefig("accuracy_pic_test2.jpg")


# Part 2

# try to distinguish different room
# use house / room as single unit?

