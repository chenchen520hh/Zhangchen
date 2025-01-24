import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from copy import deepcopy
from torch_geometric.nn import GATConv
import scipy.sparse as sp
import argparse
import os

# Argument parsing (same as before)
from torch_geometric.utils import dense_to_sparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fold', type=int, default=0, help='Number of folds')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.6)
args = parser.parse_args()
fold = args.fold
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 5000
BATCH_SIZE = 64
n = 708
m = 1512
node_num = 2220
dropout = 0.5
lr = 0.01
weight_decay = 1e-3
in_features = 2220
out_features = 16
N_HID = 32
adjust_p_neighbors_parameters = True


def normalization(adj):
    adj_ = adj + np.eye(node_num, node_num)  # A + I
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

edge_index_temp=sp.coo_matrix(A_laplacians)
edge_weight= deepcopy(edge_index_temp.data)  #边权
edge_weight=torch.FloatTensor(edge_weight).to(device)#将numpy转为tensor 我们要利用的边权
edge_index= np.vstack((edge_index_temp.row, edge_index_temp.col))#提取的边[2,num_edges]
edge_index=torch.LongTensor(edge_index).to(device) #将numpy转为tensor 我们要利用的边的index

class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.6, heads=4):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(input_dim, output_dim, heads=heads, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = self.gat_conv(x, edge_index, edge_weight)
        x = torch.sigmoid(x)
        return x


class AE(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim, dropout=0.5, heads=8):
        super(AE, self).__init__()
        self.gat1 = GATLayer(feat_dim, hid_dim, dropout=dropout, heads=heads)
        self.gat2 = GATLayer(hid_dim * heads, out_dim, dropout=dropout, heads=1)
        self.gat3 = GATLayer(hid_dim * heads, out_dim, dropout=dropout, heads=1)
        self.dc = InnerProductDecoder(dropout, act=torch.sigmoid)

    def encoder(self, x, edge_index, edge_weight):
        x1 = self.gat1(x, edge_index, edge_weight)
        h = self.gat2(x1, edge_index, edge_weight)
        std = self.gat3(x1, edge_index, edge_weight)
        return h, std

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index, edge_weight):
        mu, logvar = self.encoder(x, edge_index, edge_weight)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, self.dc(z)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act  # 默认使用 sigmoid 激活函数

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))  # 这里已经通过 sigmoid 激活了 adj
        return adj


def compute_loss_para(adj,positive_weight_factor):
    #计算正样本的权重
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # 放大正样本的权重
    pos_weight *= positive_weight_factor
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

def train(model, x, edge_index, edge_weight, adj1, epochs, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        mu, logvar, z, recon_adj = model(x, edge_index, edge_weight)
        weight_tensor, norm = compute_loss_para(adj1,positive_weight_factor=15.0)
        loss = loss_function(recon_adj, adj1, mu, logvar, norm, weight_tensor)
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch} | Loss: {loss.item():.4f}')
    return model


A_laplacians = torch.from_numpy(A_laplacians).float().to(device)
X = torch.from_numpy(X).float().to(device)
A = torch.from_numpy(A).float().to(device)
# 确保 edge_index 和 edge_weight 在正确的设备上
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

ae_model = AE(in_features, N_HID, out_features, dropout=dropout, heads=8).to(device)
trained_ae = train(ae_model, X, edge_index, edge_weight,A, epochs, lr, weight_decay)
ae_model.eval()
with torch.no_grad():
    mu, logvar, z, recon_adj = ae_model(X, edge_index,edge_weight)
    z = z.cpu().numpy()
np.savetxt('CL-result-of-VGAE/embedding--' + str(fold) + '.txt', z)
print('Training completed.')
