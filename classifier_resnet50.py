import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse
import random
import sys

# import time
import datetime

import torch
from torch.utils.data import Dataset
# from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
import numpy as np
# from torchvision.io import read_image
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms

import torch.nn as nn
# import torch.nn.functional as F

# resnet50 pretrained model import
from torchvision.models import resnet50, ResNet50_Weights

# TensorBoard import
from torch.utils.tensorboard import SummaryWriter

# classification report to print f1 score for each label
from sklearn.metrics import classification_report

# focal loss package
# sys.path.append("/home/ubuntu/pytorch-multi-class-focal-loss")
from focal_loss import focal_loss, FocalLoss

# run tensorboard: tensorboard --logdir=runs

# labels

label_list = ["closet", "bathroom", "bedroom", "living room", "kitchen", "dining room", "garage", "laundry", "hallway", 
"stair", "family room", "breakfast nook", "pantry", "loft", "patio", "office", "doorway", 
"basement", "utility", "balcony", "attic", "yard", "other"]

# custom dataset + dataloader

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
        # pano = pano.resize((512, 256))                          # size: torch.Size([3, 256, 512])
        pano = pano.resize((256, 128))                          # size: torch.Size([3, 128, 256])
        if self.transform:
            pano = self.transform(pano)
            # print(pano)
            if self.train:
                pano = np.roll(pano, random.randrange(0, 127, 1), axis=2) 
        if self.target_transform:
            label = self.target_transform(label)

        return pano, label
                            
# get label from file

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

def alpha_calculator(panos):
    count = [0] * len(label_list)
    for i, item in enumerate(panos):
        count[label_list.index(item[1])] += 1
    weights = np.asarray(count)
    weights = 1 / weights
    weights /= weights.sum()
    weights *= 1000
    # print(weights.dtype)
    # print(weights)
    print("alpha set")
    return weights

parser = argparse.ArgumentParser(description="Partition Zillow Indoor Dataset (ZInD)")

# parser.add_argument(
#     "--input", "-i", help="Input json file contains all the home tours.", required=True
#     # nargs='+'
# )

# parser.add_argument(
#     "--dataset", "-d", help="Input the directory of dataset", required=True
#     # nargs='+'
# )

# parser.add_argument(
#     "--partition", "-p", help="Input data partition."
# )

parser.add_argument(
    "--mode", "-m", help="Trainning mode. T: train, V: val, E: test", nargs='*'
    # nargs='+'
)

parser.add_argument(
    "--model", "-M", help="Input trained model", nargs='*'
    # nargs='+'
)

# parser.add_argument(
#     "--output", "-o", help="Output folder where zind_partition.json will be saved to", required=True
# )

args = parser.parse_args()
input_file = "./stats.json"
dataset_dir = "/CGVLAB3/datasets/tingwei/zind_data/"
partition_set = "./zind_partition.json"
trainning_mode = args.mode
trained_model = args.model

mode_train_bool = False
mode_val_bool = False
mode_test_bool = False

# print(trainning_mode)

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

# print(partition_data)

train_panos = PanoLabelGetter(houses_data, partition_data, "train")               # total: 67448 panos
val_panos = PanoLabelGetter(houses_data, partition_data, "val")
test_panos = PanoLabelGetter(houses_data, partition_data, "test")
# s = alpha_calculator(train_panos)
# exit()

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

transform = transforms.Compose(
    [
        # transforms.RandomGrayscale(p = 0.2),
        # transforms.RandomRotation(degrees=90),
        # transforms.RandomAffine(degrees=90, translate=(0.1, 0.3), scale=(0.5,1)),
        transforms.ColorJitter(brightness=[0.35, 1.0]),
        transforms.ToTensor(),
    ]
)


label_identifier_train = CustomZinDLabelParser(
    pano_pairs = train_panos,
    data_dir = dataset_dir,
    # transform = transforms.ToTensor(),
    transform = transform,
    train = True
)

label_identifier_val = CustomZinDLabelParser(
    pano_pairs = val_panos,
    data_dir = dataset_dir,
    # transform = transforms.ToTensor()
    transform = transform
)

label_identifier_test = CustomZinDLabelParser(
    pano_pairs = test_panos,
    data_dir = dataset_dir
)

# -------- print test --------

# train_features, train_labels = next(iter(label_identifier_train))
# img = train_features[0].squeeze()

# need to swap axes for plt to print (3, 256, 512 -> 512, 256, 3)

# img = img.swapaxes(0, 1)
# img = img.swapaxes(1, 2)

# print(type(img))
# plt.imshow(img)
# plt.show(block=True)

# -------- parameters --------

batch_size = 32
shuffle_train = True
num_workers = 8
epochs = 250
val_freq = 10
augmentation = "numpy.roll + colorJitter"

label_dataloader_train = DataLoader(label_identifier_train, batch_size = batch_size, shuffle = shuffle_train, num_workers = num_workers)
label_dataloader_val = DataLoader(label_identifier_val, batch_size = batch_size, shuffle = False)

# -------- neural network --------

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # image shape: 3 * 256 * 512, color 3 channels
#         # Conv2d: (input_channels, output_channels, kernel_size)
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         # current shape: 6 * 252 * 508
#         # Maxpool: (kernel_size, stride)
#         # take max in kernel * kernel, move stride 
#         self.pool = nn.MaxPool2d(2, 2)
#         # current shape: 6 * 126 * 254
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # current shape: 16 * 122 * 250 
#         # max pooling twice -> 16 * 61 * 125
#         self.fc1 = nn.Linear(16 * 61 * 125, 240)
#         self.fc2 = nn.Linear(240, 84)
#         self.fc3 = nn.Linear(84, 23)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# -------- append layer definition --------

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


# net = Net()

# use pre-trained model

pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
net = ExtraLayerNet(pretrained_model=pretrained)
# print(net)



# tensorboard initialize
ts_writer = SummaryWriter()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

if mode_test_bool:
    net.load_state_dict(torch.load("./zillow_net_50epochs_resnet50.pth"))
    net.eval()
    net.cuda()
    correct = 0
    total = 0
    final_arr_predict = []
    final_arr_real = []

    # nopl = False
    # norl = False

    # try:
    #     with open( "predict_list.txt" , 'r') as pl:
    #         final_arr_predict = pl
    # except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
    #     nopl = True
    
    # try:
    #     with open( "real_list.txt" , 'r') as rl:
    #         final_arr_predict = rl
    # except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
    #     norl = True

    # if nopl and norl:

    final_arr_predict_file = "predict_list.txt"
    final_arr_real_file = "real_list.txt"
    fp = open(final_arr_predict_file, 'w')
    fr = open(final_arr_real_file, 'w')

    with torch.no_grad():
        for data in label_dataloader_val:
            images, labels = data[0].cuda(), data[1].cuda()
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            final_arr_predict += predicted
            final_arr_real += labels
    accuracy = 100.0 * (float)(correct) / (float)(total)

    final_arr_predict = torch.tensor(final_arr_predict, device='cpu')
    final_arr_real = torch.tensor(final_arr_real, device='cpu')
    # output_str = f'Accuracy of the network on the epoch no. {epoch + 1}: {accuracy} %'
    # print(output_str, file=fv)
    # correct_list.append(accuracy)
    # ts_writer.add_scalar("accuracy per 2 epoch", accuracy, epoch + 1)
    
    # print val time
    # nowTime = datetime.datetime.now()
    # output_str = f"val # {epoch + 1} done, {nowTime}"
    # print(output_str)
    # print(output_str, file=fs)

    # test model


    val_label_list = [name for i, name in enumerate(label_list) if i in final_arr_real]
    print(val_label_list)
    # report = classification_report(final_arr_real, final_arr_predict, target_names=label_list)
    report = classification_report(final_arr_real, final_arr_predict)
    # report = classification_report(final_arr_real, final_arr_predict, labels=label_list)
    print(report)
    # print(report, file="testing_status.txt")

losses = []
correct_list = []

if mode_train_bool:

    net.cuda()          # train on GPU

    # -------- start to train --------

    import torch.optim as optim

    # output file
    out_path_train = "train_result.txt"
    out_path_val = "val_result.txt"
    out_path_status = "training_status.txt"
    ft = open(out_path_train, 'w')
    fv = open(out_path_val, 'w')
    fs = open(out_path_status, 'w')

    # use focal loss
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(
                    alpha=torch.tensor(alpha_calculator(train_panos)).float().cuda(),               # set alpha for each class
                    gamma=2.0
                )
    optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
    # add LR decay
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100, 160], gamma=0.1)


    nowTime = datetime.datetime.now()
    
    output_str = f"parameters: \nbatch_size = {batch_size}\nshuffle_train = {shuffle_train}\nnum_workers = {num_workers}\nepoch = {epochs}\naugmentation = {augmentation}"
    print(output_str)
    print(output_str, file=fs)

    output_str = f"Start training, {nowTime}"
    print(output_str)
    print(output_str, file=fs)

    loss_freq = 50

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(label_dataloader_train, 0):
            # print(data[1])
            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(f"last_lr sche: {scheduler.get_last_lr()}")

            # print statistics
            running_loss += loss.item()
            # loss.item()
            if i % loss_freq == loss_freq - 1 or i == len(label_dataloader_train) - 1:
                running_loss /= (i % loss_freq + 1)
                output_str = f'[{epoch + 1}] [data:{i+1}] loss: {running_loss:.6f}'
                print(output_str, file=ft)
                losses.append(running_loss)
                ts_writer.add_scalar(f"Loss per {loss_freq} mini_batch", running_loss, epoch * len(label_dataloader_train) + (i + 1))
                running_loss = 0.0

        # get current time
        nowTime = datetime.datetime.now()
        output_str = f"epoch # {epoch + 1} done, {nowTime}"
        print(output_str)
        print(output_str, file=fs)

        # validation
        if epoch % val_freq == val_freq - 1 and mode_val_bool:
            correct = 0
            total = 0
            final_arr_predict = []
            final_arr_real = []
            with torch.no_grad():
                for data in label_dataloader_val:
                    images, labels = data[0].cuda(), data[1].cuda()
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if epoch == epochs - 1:
                        final_arr_predict += predicted
                        final_arr_real += labels
            accuracy = 100.0 * (float)(correct) / (float)(total)
            output_str = f'Accuracy of the network on the epoch no. {epoch + 1}: {accuracy} %'
            print(output_str, file=fv)
            correct_list.append(accuracy)
            ts_writer.add_scalar(f"accuracy per {val_freq} epoch", accuracy, epoch + 1)
            
            # print val time
            nowTime = datetime.datetime.now()
            output_str = f"val # {epoch + 1} done, {nowTime}"
            print(output_str)
            print(output_str, file=fs)
        
        # save & val model
        if epoch == epochs - 1:
            # if trained_model:
            #     model_path = f'./zillow_net_{epoch + 51}epochs_resnet50.pth'
            # else:
            model_path = f'./classifier_{epoch + 1}epochs_resnet50.pth'
            torch.save(net.state_dict(), model_path)
            final_arr_predict = torch.tensor(final_arr_predict, device='cpu')
            final_arr_real = torch.tensor(final_arr_real, device='cpu')
            report = classification_report(final_arr_real, final_arr_predict)
            print(report)
            print(report, file=fs)

        
        scheduler.step()


    ts_writer.flush()
    ts_writer.close()
    print('Finished Training')
    print('Finished Training', file=fs)




# -------- print loss curve --------

# use train_result.txt to test draw

# epoch_loss = []
# epoch_size = 845
loss_temp = 0.0
if not mode_train_bool and not mode_test_bool: 
    print("print only")
    train_result = open('train_result.txt')
    val_result = open('val_result.txt')
    i = 0
    for n, line in enumerate(train_result):
        # if "[1]" in line:
        # losses.append((float)(line.split(" ")[3].split("\n")[0]))
        loss_temp = ((float)(line.split(" ")[3].split("\n")[0]))
        # loss_temp *= 50
        # loss_temp /= (float)(((int)(line.split(" ")[1].split(":")[1].split("]")[0]) - 1) % 50 + 1) 
        losses.append(loss_temp)
        ts_writer.add_scalar(f"Loss per {50} mini_batch", loss_temp, (n + 1) * 50)
        # i += 1
        
    i = 0
    for n, line in enumerate(val_result):
        accuracy = (float)(line.split(" ")[9])
        correct_list.append(accuracy)
        ts_writer.add_scalar("accuracy per 2 epoch", accuracy, n * 2 + 2)

    ts_writer.flush()
    ts_writer.close()

# print(losses)

exit()

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

x2 = range(0, epochs, 2)
y2 = correct_list
plt.figure()
plt.plot(x2, y2)
plt.title('accuracy per 2 epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(60, 80)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
# plt.show()
plt.savefig("accuracy_pic_test.jpg")
# plt.ylim(40, 60)
# plt.savefig("accuracy_pic_test2.jpg")




