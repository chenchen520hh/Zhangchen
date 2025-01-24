import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader

from my_function import *
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np
import copy
import argparse
import torch.nn as nn
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fold', type=int, default=9, help='Number of folds')
args = parser.parse_args()
fold = args.fold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 2214#n是药物数据
m = 1968#m是靶点数量

# 读取数据
index_1 = np.loadtxt('divide_test1/index_1.txt')#10折交叉验证中1的编号
index_0 = np.loadtxt('divide_test1/index_0.txt')#10折交叉验证中0的编号
index = np.hstack((index_1, index_0)) #合并正负样本的编号
#加载特征向量和标签
drug_feature = np.loadtxt('test_result_of_VGAE/embedding'+str(fold)+'.txt')[0:n, :]#前n行(0到n-1行)是学到的药物的特征向量
target_feature = np.loadtxt('test_result_of_VGAE/embedding'+str(fold)+'.txt')[n:, :]#后面是靶标的特征向量
label=np.loadtxt('dataset2/mat_data/mat_drug_protein.txt') #标签矩阵
# 获得训练集与测试集的indexmax = {float64: ()} 0.03173387795686722
idx = copy.deepcopy(index)  #深拷贝的意思就是对idx的任何操作都不会影响index
test_index = copy.deepcopy(idx[fold]) #提取特定折的测试集索引

idx=np.delete(idx,fold,axis=0) #删除当前折的索引 剩下的就是训练集的索引
train_index = idx.flatten() #flatten()将多维数组转换为一维数组
insersection=np.intersect1d(test_index,train_index)#查看训练集和测试集是否有交集
#print(insersection)
# ！！！注意这里！！！！测试数据不能打乱，训练数据可以打乱
np.random.seed(10)
np.random.shuffle(train_index)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out
# 获得 （测试集 与 训练集） 的 （输入向量 与 标签）
# 此时test_index和train_index都是一维的
test_input, test_output = get_feature(drug_feature, target_feature, label, test_index, n, m)
train_input, train_output = get_feature(drug_feature, target_feature, label, train_index, n, m)#原来不用trick的一行代码

# 设置参数
input_size = train_input.shape[1]
hidden_size = 100  # 可以根据需要调整
num_classes = 1

model = MLP(input_size, hidden_size, num_classes)
model = model.to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据标准化
scaler = StandardScaler()
train_input = scaler.fit_transform(train_input)
test_input = scaler.transform(test_input)

# 转换为torch张量
train_input = torch.tensor(train_input, dtype=torch.float32)
train_output = torch.tensor(train_output, dtype=torch.float32)
test_input = torch.tensor(test_input, dtype=torch.float32)
test_output = torch.tensor(test_output, dtype=torch.float32)

# 数据加载器
train_data = TensorDataset(train_input, train_output.view(-1, 1))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 训练模型
model.train()
for epoch in range(1000):  # 迭代次数可以调整
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    y_pred = model(test_input.to(device)).cpu().numpy()

# 阈值处理
y_pred_label = (y_pred > 0.5).astype(int)


# 评价指标
print('The AUC of prediction is:', roc_auc_score(test_output, y_pred))
print('The AUPR of prediction is:', average_precision_score(test_output, y_pred))
print('Precision:', precision_score(test_output, y_pred_label))
print('F1 Score:', f1_score(test_output, y_pred_label))
print('MCC:', matthews_corrcoef(test_output, y_pred_label))