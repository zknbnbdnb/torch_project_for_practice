from pathlib import Path
from numpy.ma.core import get_data
import requests
from matplotlib import pyplot as pyplot
import numpy as np
import pickle
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = Path('D:\pytorch\pytorch项目')

PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = 'https://deeplearning.net/data/mnist'

FILENAME = 'mnist.pkl.gz'

# if not (PATH / FILENAME).exists():
#     content = requests.get(URL + FILENAME).content
#     (PATH / FILENAME).open('wb').write(content)

with gzip.open((PATH / FILENAME).as_posix(), 'rb')as f:
    ((x_train, y_train), (x_valid, y_valid),
     _) = pickle.load(f, encoding='latin1')

pyplot.imshow(x_train[0].reshape((28, 28)), cmap='gray')
# pyplot.show()
# output:(50000, 784) 784 = 1* 28 * 28 (c, w, h)
print(x_train.shape)

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.max(), y_train.min())

# output:
# tensor([[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]]) tensor([5, 0, 4,  ..., 8, 4, 8])

loss_func = F.cross_entropy


def model(xb):
    return xb.mm(weights) + bias


bs = 64
xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype=torch.float,  requires_grad=True)
bs = 64
bias = torch.zeros(10, requires_grad=True)

print(loss_func(model(xb), yb))


class Mnist_NN(nn.Module):
    """Some Information about Mnist_NN"""

    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


net = Mnist_NN()
# print(net)
# output:
# Mnist_NN(
#   (hidden1): Linear(in_features=784, out_features=128, bias=True)
#   (hidden2): Linear(in_features=128, out_features=256, bias=True)
#   (out): Linear(in_features=256, out_features=10, bias=True)
# )

# for name, parameter in net.named_parameters():
#     print(name, parameter, parameter.size())
# output:
# hidden1.weight Parameter containing:
# tensor([[ 0.0083, -0.0016,  0.0156,  ...,  0.0095,  0.0153, -0.0202],
#         [ 0.0006,  0.0351,  0.0134,  ..., -0.0326, -0.0269, -0.0251],
#         [ 0.0076, -0.0080, -0.0038,  ...,  0.0044, -0.0283,  0.0187],
#         ...,
#         [ 0.0227,  0.0333, -0.0184,  ...,  0.0275, -0.0242,  0.0033],
#         [ 0.0190, -0.0060, -0.0252,  ..., -0.0037, -0.0310, -0.0153],
#         [-0.0215, -0.0251, -0.0158,  ...,  0.0230, -0.0309, -0.0097]],
#        requires_grad=True) torch.Size([128, 784])
# hidden1.bias Parameter containing:
# tensor([-0.0342, -0.0338, -0.0303, -0.0084,  0.0288, -0.0082,  0.0332, -0.0038,
#          0.0350,  0.0039,  0.0354, -0.0305, -0.0304,  0.0162,  0.0029,  0.0222,
#          0.0049,  0.0221, -0.0277,  0.0329, -0.0022, -0.0201,  0.0254, -0.0098,
#         -0.0070, -0.0024,  0.0157,  0.0324,  0.0271, -0.0113,  0.0087,  0.0104,
#          0.0167,  0.0060, -0.0213, -0.0049,  0.0048, -0.0214,  0.0037,  0.0344,
#         -0.0050, -0.0157, -0.0354,  0.0228, -0.0171,  0.0201, -0.0297, -0.0210,
#         -0.0252, -0.0343,  0.0241,  0.0274, -0.0215, -0.0095, -0.0044,  0.0347,
#          0.0203,  0.0352,  0.0019, -0.0170,  0.0194,  0.0185,  0.0048, -0.0143,
#         -0.0027,  0.0135, -0.0342, -0.0038,  0.0148,  0.0016, -0.0231, -0.0183,
#         -0.0083,  0.0178, -0.0021,  0.0308, -0.0035,  0.0030,  0.0020,  0.0049,
#          0.0276,  0.0091,  0.0285,  0.0092,  0.0296, -0.0010,  0.0354, -0.0050,
#         -0.0292, -0.0207, -0.0288, -0.0164, -0.0184, -0.0207,  0.0093,  0.0332,
#          0.0080,  0.0332, -0.0098,  0.0157,  0.0235,  0.0200,  0.0187,  0.0182,
#         -0.0033,  0.0277, -0.0296,  0.0059, -0.0125, -0.0145, -0.0324, -0.0317,
#          0.0283, -0.0120,  0.0295,  0.0100,  0.0208,  0.0070, -0.0275, -0.0266,
#         -0.0131, -0.0288,  0.0174, -0.0070, -0.0126,  0.0125,  0.0234,  0.0074],
#        requires_grad=True) torch.Size([128])
# hidden2.weight Parameter containing:
# tensor([[ 0.0481,  0.0633, -0.0423,  ...,  0.0283, -0.0068, -0.0709],
#         [-0.0082, -0.0844, -0.0331,  ...,  0.0676, -0.0470,  0.0013],
#         [-0.0122,  0.0663,  0.0457,  ..., -0.0492,  0.0634, -0.0748],
#         ...,
#         [ 0.0357, -0.0186, -0.0871,  ..., -0.0814, -0.0726,  0.0402],
#         [ 0.0700, -0.0286, -0.0287,  ...,  0.0381,  0.0144,  0.0081],
#         [ 0.0153, -0.0276,  0.0823,  ...,  0.0626, -0.0875,  0.0111]],
#        requires_grad=True) torch.Size([256, 128])
# hidden2.bias Parameter containing:
# tensor([ 0.0844,  0.0162, -0.0670, -0.0431,  0.0265,  0.0309,  0.0453, -0.0766,
#         -0.0731, -0.0207,  0.0492, -0.0744,  0.0138, -0.0533,  0.0266,  0.0230,
#          0.0849, -0.0424, -0.0365,  0.0266,  0.0466,  0.0036,  0.0363, -0.0775,
#          0.0579, -0.0841,  0.0420,  0.0719,  0.0575,  0.0448,  0.0199,  0.0209,
#         -0.0838,  0.0333,  0.0214,  0.0182,  0.0671, -0.0472,  0.0586, -0.0623,
#          0.0198,  0.0043,  0.0199, -0.0057,  0.0322, -0.0425, -0.0186,  0.0039,
#         -0.0374, -0.0743,  0.0243, -0.0458, -0.0108,  0.0330,  0.0562, -0.0779,
#          0.0325, -0.0339, -0.0310,  0.0292,  0.0489,  0.0274,  0.0829,  0.0751,
#          0.0232,  0.0544, -0.0635,  0.0237,  0.0846,  0.0048,  0.0410,  0.0203,
#          0.0117,  0.0144, -0.0765,  0.0227,  0.0592,  0.0801, -0.0357, -0.0830,
#          0.0074, -0.0167,  0.0433,  0.0107,  0.0811, -0.0271, -0.0257,  0.0785,
#          0.0702,  0.0069, -0.0179,  0.0565, -0.0858,  0.0370, -0.0772,  0.0057,
#         -0.0198,  0.0537, -0.0756, -0.0496,  0.0377,  0.0861,  0.0023, -0.0597,
#          0.0304, -0.0706, -0.0209, -0.0666, -0.0862, -0.0815,  0.0544, -0.0689,
#          0.0329,  0.0690,  0.0611,  0.0832,  0.0644,  0.0360, -0.0754, -0.0048,
#          0.0294,  0.0302,  0.0439,  0.0490,  0.0164, -0.0303, -0.0574,  0.0322,
#         -0.0622, -0.0148, -0.0694, -0.0491, -0.0371,  0.0621, -0.0843, -0.0237,
#          0.0312, -0.0873,  0.0561,  0.0656,  0.0052, -0.0018, -0.0784,  0.0127,
#          0.0565, -0.0567,  0.0862, -0.0322, -0.0457, -0.0761,  0.0814,  0.0181,
#          0.0083,  0.0513,  0.0114,  0.0713, -0.0721, -0.0865, -0.0572, -0.0741,
#         -0.0249, -0.0732, -0.0649, -0.0580, -0.0681, -0.0473,  0.0286,  0.0520,
#         -0.0180,  0.0273,  0.0648,  0.0705,  0.0546, -0.0102, -0.0260, -0.0428,
#         -0.0618,  0.0468, -0.0081, -0.0674, -0.0087, -0.0236,  0.0087,  0.0610,
#          0.0494,  0.0866, -0.0466,  0.0466,  0.0455, -0.0859,  0.0749,  0.0588,
#          0.0290, -0.0605, -0.0105, -0.0873, -0.0706, -0.0678, -0.0162,  0.0811,
#         -0.0005, -0.0735, -0.0002, -0.0337, -0.0215, -0.0087, -0.0296, -0.0844,
#         -0.0851,  0.0861,  0.0319,  0.0417, -0.0661, -0.0651, -0.0416,  0.0644,
#          0.0082, -0.0763, -0.0580,  0.0513, -0.0310, -0.0848, -0.0753, -0.0602,
#          0.0545, -0.0155, -0.0705,  0.0284, -0.0721, -0.0781,  0.0066,  0.0119,
#          0.0499,  0.0147,  0.0406,  0.0142,  0.0146, -0.0763, -0.0156, -0.0198,
#          0.0442,  0.0409, -0.0003, -0.0291, -0.0607, -0.0819, -0.0640, -0.0628,
#         -0.0277, -0.0703,  0.0529,  0.0202,  0.0384,  0.0683,  0.0338,  0.0357],
#        requires_grad=True) torch.Size([256])
# out.weight Parameter containing:
# tensor([[-1.5139e-02,  1.8695e-02, -2.8823e-02,  ...,  5.1267e-02,
#          -4.4238e-02,  5.6465e-02],
#         [-3.4239e-02, -2.8567e-02,  1.7235e-02,  ..., -7.1350e-03,
#          -2.2321e-02, -5.7858e-02],
#         [-4.2465e-02,  4.8588e-02, -4.4067e-02,  ...,  1.9611e-02,
#          -8.0302e-03,  4.9489e-02],
#         ...,
#         [ 5.7825e-02, -2.3490e-03,  2.2900e-02,  ..., -2.8915e-02,
#           5.9604e-02, -5.4130e-02],
#         [-6.2278e-02, -3.6162e-02, -3.8026e-02,  ...,  4.9024e-02,
#          -3.7363e-02,  3.4999e-02],
#         [-8.4750e-05,  1.6231e-04,  2.2459e-02,  ..., -1.4192e-03,
#           4.8301e-02, -4.6313e-02]], requires_grad=True) torch.Size([10, 256])
# out.bias Parameter containing:
# tensor([-0.0050,  0.0043, -0.0466,  0.0272, -0.0279, -0.0193,  0.0391, -0.0315,
#         -0.0239,  0.0370], requires_grad=True) torch.Size([10])

train_ds = TensorDataset(x_train.requires_grad_(), y_train.requires_grad_())
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

valid_ds = TensorDataset(x_valid.requires_grad_(), y_valid.requires_grad_())
valid_dl = DataLoader(valid_ds, batch_size=128, shuffle=False)


def get_data(train_ds, valid_ds):
    return(
        DataLoader(train_ds, batch_size=64, shuffle=True),
        DataLoader(valid_ds, batch_size=128, shuffle=False),
    )


def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        # 一般再训练模型上加上model.train()， 这样就会正常使用BN和DropOut
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        # 测试的时候一般选择model.eval()， 这样不会使用BN和DropOut
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums) / np.sum(nums))
        print('当前step：'+str(step), '验证集损失：'+str(val_loss))


def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


train_dl, valid_dl = get_data(train_ds, valid_ds)
model, opt = get_model()
fit(30, model, loss_func, opt, train_dl, valid_dl)

# output:
# 当前step:0 验证集损失：2.282871880722046
# 当前step:1 验证集损失：2.2553405284881594
# 当前step:2 验证集损失：2.212825806427002
# 当前step:3 验证集损失：2.1420564094543457
# 当前step:4 验证集损失：2.024079023742676
# 当前step:5 验证集损失：1.8402583400726318
# 当前step:6 验证集损失：1.5972708093643189
# 当前step:7 验证集损失：1.3420549598693847
# 当前step:8 验证集损失：1.1238106727600097
# 当前step:9 验证集损失：0.9571563722610473
# 当前step:10 验证集损失：0.833206335067749
# 当前step:11 验证集损失：0.7411214539527893
# 当前step:12 验证集损失：0.6710761416435241
# 当前step:13 验证集损失：0.6161400938034057
# 当前step:14 验证集损失：0.5724429847717285
# 当前step:15 验证集损失：0.5371728239536285
# 当前step:16 验证集损失：0.5079221643447877
# 当前step:17 验证集损失：0.4832162464141846
# 当前step:18 验证集损失：0.4629469300746918
# 当前step:19 验证集损失：0.4450542440891266
# 当前step:20 验证集损失：0.4298845146656036
# 当前step:21 验证集损失：0.41687224960327146
# 当前step:22 验证集损失：0.4055648094415665
# 当前step:23 验证集损失：0.3952901126384735
# 当前step:24 验证集损失：0.3864197319984436
