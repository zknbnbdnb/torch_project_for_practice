import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.modules import loss
from torch.nn.modules.activation import Sigmoid
import torch.optim as optim
import datetime
from sklearn import preprocessing
# import warnings
# warnings.filterwarnings('ignore')

# 文件处理
features = pd.read_csv('D:\pytorch\pytorch_tutorial/010_015：神经网络实战分类与回归任务\神经网络实战分类与回归任务/temps.csv')
print(features)

print('数据维度：', features.shape)

# 处理时间
years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Python time strptime() 函数根据指定的格式把一个时间字符串解析为时间元组。
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

print(dates[:5])

# 开始画图
plt.style.use('fivethirtyeight')
# 可以plt.subplots()一次制作所有子图，然后将子图的图形和轴（复数轴）作为元组返回。可以将图形理解为在其中绘制草图的画布。
# 而plt.subplot()如果要单独添加子图，则可以使用。它仅返回一个子图的轴。
# plt.subplots()首选，因为它为您提供了更轻松的选项来直接自定义整个图形
# 使用时plt.subplot()，必须为每个轴分别指定，这可能会很麻烦。
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10))
fig.autofmt_xdate(rotation = 45)
# 创建四幅图 行两张，列两张
ax1.plot(dates, features['actual']) # 读入dates数据
ax1.set_xlabel(''); ax1.set_ylabel('Temperate'); ax1.set_title('Max Temp')

ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax1.set_ylabel('Temperate'); ax1.set_title('Previous Max Temp')

ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperate'); ax3.set_title('Two Days Prior Max Temp')

ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperate'); ax4.set_title('Friend Estimate')

# tight_layout会自动调整子图参数，使之填充整个图像区域。这是个实验特性，可能在一些情况下不工作。
plt.tight_layout(pad = 2)
plt.show()

# 独热编码, 处理星期几
features = pd.get_dummies(features)
print(features.head(5))

labels = np.array(features['actual'])
# axis = 1 指定删除相关的列
features = features.drop('actual', axis = 1)

features_list = list(features.columns)
features = np.array(features)
print(features.shape)

# 数据标准化，使收敛速度加快，收敛损失减少
# preprocessing.StandardScaler().fit_transform：不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
input_features = preprocessing.StandardScaler().fit_transform(features)
'''print(input_features)

# 构建模型
x = torch.tensor(input_features, dtype = float)
y = torch.tensor(labels, dtype = float)

#  权重参数初始化

weights = torch.randn((14, 128), dtype = float, requires_grad=True)
biases = torch.randn(128, dtype = float, requires_grad = True)
weights2 = torch.randn((128, 1), dtype = float, requires_grad = True)
biases2 = torch.randn(1, dtype = float, requires_grad = True)

learning_rate = 0.001 
losses = [] 

for i in range(1000):
    # 计算隐层 [348, 14] * [14, 128] = [348, 128] 转化为128个隐层特征
    # biases是偏置参数， 均对隐层进行微调
    # .mm为矩阵乘法
    hidden = x.mm(weights) + biases
    # 加入激活函数
    hidden = torch.relu(hidden)
    # 预测结果
    # [348, 128] * [128， 1] = [128， 1]
    # 回归模型
    predictions = hidden.mm(weights2) + biases2
    # 计算损失
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())
    # 打印损失值
    if i % 100 == 0:
        print('loss:', loss)
    # 反向传播计算
    loss.backward()
    # 更新参数
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)
    # 每次迭代更新梯度
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()'''

input_size = input_features.shape[1] # print 14
hidden_size = 128
output_size = 1 
batch_size = 16 
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(), 
    torch.nn.Linear(hidden_size, output_size),
)

cost = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr = 0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

losses = []  
for i in range(1000):
    batch_loss = []  
    # MINI-batch 方法进行训练
    for start in range(0, len(input_features), batch_size):
        # 每次加载规定的batch_size
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype = torch.float, requires_grad = True)
        yy = torch.tensor(labels[start:end], dtype = torch.float, requires_grad = True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        # 网络套路三部曲，清零梯度， 反向传播， 下一步
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        batch_loss.append(loss.data.numpy())
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i + 100, np.mean(batch_loss))

# 设置测试集
x = torch.tensor(input_features, dtype = torch.float)
predict = my_nn(x).data.numpy()

# 测试集和真实数据加载到DataFrame，更容易让matplotlib进行绘制操作
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

true_data = pd.DataFrame(data = {'date' : dates, 'actual': labels})

months = features[:, features_list.index('month')]
days = features[:, features_list.index('day')]
years = features[:, features_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame({'date': test_dates, 'prediction': predict.reshape(-1)})

# 绘制图像观察拟合程度
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction') 
plt.xticks(rotation = '60');
plt.legend()

plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');
plt.show()


