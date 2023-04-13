import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

input_size = 28
num_classes = 10
num_epochs = 10
batch_size = 64

train_dataset = datasets.MNIST(
    root='D:\pytorch\pytorch项目\mnist', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='D:\pytorch\pytorch项目\mnist', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class CNN(nn.Module):
    """Some Information about CNN"""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    right = pred.eq(labels.data.view_as(pred)).sum()
    return right, len(labels)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = CNN()
net = net.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

# 开始训练循环
for epoch in range(num_epochs):
    # 当前epoch的结果保存下来
    train_rights = []

    for batch_idx, input in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        net.train()

        data, target = input
        data = data.to(device)
        target = target.to(device)
        output = net(data)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        right = accuracy(output, target)
        train_rights.append(right)

        if batch_idx % 100 == 0:

            net.eval()
            val_rights = []

            for input in test_loader:
                data, target = input
            data = data.to(device)
            target = target.to(device)
            output = net(data)

            right = accuracy(output, target)
            val_rights.append(right)

            # 准确率计算
            train_r = (sum([tup[0] for tup in train_rights]),
                       sum([tup[1] for tup in train_rights]))

            val_r = (sum([tup[0] for tup in val_rights]),
                     sum([tup[1] for tup in val_rights]))

            print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].cpu().numpy() / train_r[1],
                       100. * val_r[0].cpu().numpy() / val_r[1]))
