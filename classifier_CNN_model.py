import torch
from torch import nn
import torch.nn.functional as F

import torch.utils
import torch.utils.data
import torchvision as tv

import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image
import pickle


class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1:str, path_dir2:str):
        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)
    
    def __getitem__(self, idx):

        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0

        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA) # интерполяция для того чтобы изображение было более точным с его уменьшением, для расстяжения используем линейную или кубическую

        img = img.transpose((2, 0, 1))

        t_img = torch.from_numpy(img)

        t_class_id = torch.tensor(class_id)

        return {'img': t_img, 'label': t_class_id}
    
        # return img, class_id

train_dogs_path = './archive-3/dataset/training_set/dogs'
train_cats_path = './archive-3/dataset/training_set/cats'

test_dogs_path = './archive-3/dataset/test_set/dogs'
test_cats_path = './archive-3/dataset/test_set/cats'

train_ds_catsdogs = Dataset2class(train_dogs_path, train_cats_path)
test_ds_catsdogs = Dataset2class(test_dogs_path, test_cats_path)

# plt.imshow(train_ds_catsdogs[0][0])
# plt.show()

batch_size = 16
train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs,
    batch_size=batch_size,
    shuffle=1, # этот параметр нужен чтобы перемешивать данные при каждой эпохе, это помогает от переобучения
    drop_last=True # этот параметр нужен для того чтобы откидались данные которые имеют неполный батч, это нужно для того чтобы сохранить чтобы все батчи имели одну разммерность и не возникало никаких ошибок
)

test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs,
    batch_size=batch_size,
    shuffle=1,
    drop_last=False
)

"""конец раздела dataloader"""

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # maxpool для того чтобы выделить только самые важные признаки то есть после свертки он выбирает только самые важные признаки

        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(512, 40)
        self.linear2 = nn.Linear(40, 2)

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0) # 3 - количество каналов, так как у нас RGB у нас это 3. 32 - это у нас количество признаков. 3 - kernel_size это ядро свертки. stride - шаг с которым ходит наша свертка по изображению. Padding - это какое количество пикселей будет прибавляться к нашему изображению, это нужно для того чтобы изображение не уменьшалось, но в нашем случае это не страшно
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)


    def forward(self, x):
        
        # print(x.shape)
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)

        # print(out.shape)
        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)

        # print(out.shape)
        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)

        # print(out.shape)
        out = self.conv3(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv5(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.adaptivepool(out)
        out = self.flat(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out

# model = CNNModel()

# for sample in train_loader:
#     model(sample['img'])
#     break

"""конец раздела Архитектура нейронной сети"""

model = CNNModel()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(count_params(model))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999))

def accuracy(pred, label):
    """Метрика"""
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


epochs = 10

for epoch in (pbar := tqdm(range(epochs))):
    accur_val = 0
    loss_val = 0
    print(epoch)

    for sample in (pbar := tqdm(train_loader)):
        img, label = sample['img'], sample['label']
        optimizer.zero_grad()
        label = F.one_hot(label, 2).float()
        pred = model(img)

        loss = loss_fn(pred, label)

        loss.backward()
        loss_item = loss.item()
        loss_val += loss_item

        optimizer.step()

        acc_current = accuracy(pred, label)
        accur_val += acc_current
        print(loss_val / len(train_loader))
        print(accur_val / len(train_loader))

print("--------------TEST CHECK--------------")

with torch.no_grad():
    for sample in (pbar := tqdm(test_loader)):
        img, label = sample['img'], sample['label']
        label = F.one_hot(label, 2).float()
        pred = model(img)

        loss = loss_fn(pred, label)
        loss_item = loss.item()
        loss_val += loss_item

        acc_current = accuracy(pred, label)
        accur_val += acc_current
        print(loss_val / len(test_loader))
        print(accur_val / len(test_loader))


# img_path = input('Enter path to the image pls(size 28x28):')
img_path = '/Users/mac/Desktop/catdosgAI/flint.png'
(width, height) = Image.open(img_path).size

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
img = img/255.0
img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
img = img.transpose((2, 0, 1))

t_img = torch.from_numpy(img)
t_img = t_img.unsqueeze(0)

prediction = model(t_img)

print(F.softmax(prediction).detach().numpy())

if F.softmax(prediction).detach().numpy().argmax() == 0:
    print('dog')
else:
    print('cat')

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model have been successfully saved !")