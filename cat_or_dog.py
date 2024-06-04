import cv2
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import pickle


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


with open('/Users/mac/Desktop/catdosgAI/model.pkl', 'rb') as file:
    model = pickle.load(file)


img_path = '/Users/mac/Desktop/catdosgAI/photo1710500845.jpeg'
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