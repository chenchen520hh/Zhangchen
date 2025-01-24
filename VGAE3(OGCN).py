import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
from copy import deepcopy
from GR import sparse_matrix
from my_function import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fold', type=int, default=1, help='Number of folds')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2)
args = parser.parse_args()
fold = args.fold
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 5000
BATCH_SIZE = 64
n = 708
m = 1512
node_num=2220
dropout = 0.2
lr = 0.01
weight_decay = 1e-3
in_features = 2220
out_features = 16
N_HID = 32
adjust_p_neighbors_parameters = True

def normalization(adj):

    adj_ = adj + np.eye(node_num, node_num)  # A+IN
    row_sum = np.array(adj_.sum(1))  # 求度矩阵D
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # D^-1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 将一些计算得到的NAN值赋0值
    d_mat_inv_sqrt = np.mat(np.diag(d_inv_sqrt))  # 将D^-1/2对角化
    gcn_fact = d_mat_inv_sqrt * adj_ * d_mat_inv_sqrt  # 计算D^-1/2AD^-1/2

    return gcn_fact

A = np.loadtxt('divide_test/A' + str(fold) + '.txt')
X = np.loadtxt('divide_test/X' + str(fold) + '.txt')
# 构建拉普拉斯矩阵  此时的A矩阵已经包含了WNKN的消息
A_laplacians = normalization(A)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., act=nn.ReLU(), **kwargs):
        super(GraphConvolution, self).__init__()
        w = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weights = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
        self.dropout = dropout
        # self.adj = adj
        self.act = act

    def forward(self, inputs, adj):
        x = inputs
        x = torch.matmul(x, self.weights)
        # x = tf.sparse_tensor_dense_matmul(self.adj, x)
        x = torch.matmul(adj, x)
        outputs = self.act(x)
        return outputs
class GraphConv(nn.Module):
    def __init__(self, n_feat, n_class):     # 原始特征数 第一层结束特征数 最终特征数
        super(GraphConv, self).__init__()
        self.gc1 = GraphConvolution(input_dim=n_feat,output_dim=n_class,use_bias=True) #邻接矩阵中没有自连，我们还是加上自连吧

    def forward(self, x, adj): #x:结点特征矩阵 edge_index:coo格式存储的边索引 edge_weight：边权
        x = self.gc1(x,adj)
        x = torch.sigmoid(x)
        return x


# 变分图自动编码器  用于编码
class AE(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):  # feat_dim为输入特征的维度,hid_dim隐藏层特征的维度,out_dim输出特征的维度
        super(AE, self).__init__()
        # 两个卷积层
        self.conv1 = GraphConv(feat_dim, hid_dim)
        self.conv2 = GraphConv(hid_dim, out_dim)
        self.conv3 = GraphConv(hid_dim, out_dim)
        self.dc = InnerProductDecoder(dropout, act=torch.sigmoid)

    def encoder(self, x, adj):
        x1 = self.conv1(x, adj)  # 通过第一层得到中间表示 x
        h = self.conv2(x1, adj)  # 得到潜在变量的均值
        std = self.conv3(x1, adj)  # 得到潜在变量的标准差
        return h, std

    # 重参数化技巧 用于从均值和方差中生成潜在变量
    def reparameterize(self, mu, logvar):
        if self.training:  # 在训练模式下，使用标准正态分布生成的随机变量 eps，通过 eps * std + mu 计算得到重新参数化的潜在变量 z
            std = torch.exp(logvar)  # 指数函数
            eps = torch.randn_like(std)  # 返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充。
            return eps.mul(std).add_(mu)  # mul()相乘，原数不变，mul_()相乘，原数变动，add()同理
        else:
            return mu

    # 更新参数后的前向传播 先编码 在重参数化 最后解码
    def forward(self, x,adj):  # 这里g表示图的邻接矩阵
        mu, logvar = self.encoder(x,adj)  # 重新进行编码
        z = self.reparameterize(mu, logvar)  # 参数更新
        return mu, logvar, z, self.dc(z)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

def compute_loss_para(adj):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

def loss_function(recon_adj, adj, mu, logvar,norm,weight_tensor):
    bce_loss = norm*F.binary_cross_entropy(recon_adj.view(-1), adj.to_dense().view(-1), weight=weight_tensor)
    kl_loss = -0.5 / recon_adj.size(0) * torch.mean(torch.sum(1 + 2 * logvar - mu ** 2 - torch.exp(2 * logvar), 1))
    return bce_loss + kl_loss


def train(model, x, adj,adj1 ,epochs, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        mu, logvar, z, recon_adj = model(x,adj)
        weight_tensor,norm = compute_loss_para(adj1)
        loss = loss_function(recon_adj, adj1, mu, logvar,norm,weight_tensor)
        optimizer.zero_grad()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch} | Loss: {loss.item():.4f}')
    return model


A_laplacians = torch.from_numpy(A_laplacians).float().to(device)
X = torch.from_numpy(X).float().to(device)
A = torch.from_numpy(A).float().to(device)
ae_model = AE(in_features, N_HID, out_features).to(device)
trained_ae = train(ae_model, X, A_laplacians,A, epochs, lr, weight_decay)
ae_model.eval()
with torch.no_grad():
    mu, logvar, z, recon_adj = ae_model(X,A_laplacians)
    z = z.cpu().numpy()
np.savetxt('CL-result-of-VGAE/embedding12.11-ptGCN-' + str(fold) + '.txt', z)
print('Training completed.')
