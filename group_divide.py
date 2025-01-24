import random
import copy

import numpy as np

fold = 10
m=708 #708个药物
n=1512 #1512个靶点

#读取数据
SD=np.loadtxt('whole_data/multi_similarity/drug_fusion_similarity_708_708.txt')#药物相似度矩阵
ST=np.loadtxt('whole_data/multi_similarity/target_fusion_similarity_1512_1512.txt')#靶标相似度矩阵
DDI = np.loadtxt("whole_data/mat_drug_drug.txt")#药物相互作用
TTI = np.loadtxt("whole_data/mat_protein_protein.txt")#靶标相互作用
A = np.loadtxt("whole_data/mat_drug_protein.txt")    # A矩阵  原始标签数据
A_WKNKN=np.loadtxt('whole_data/multi_similarity/DTI_708_1512_WKNKN_MAX_DISCRETIZE.txt') #经过WKNKN补全后的邻接矩阵,并离散化
# 划分训练集和测试集的时候按照原始数据集划分
A_dim1 = A.flatten()                # 708*1512=1070496 展平矩阵
i = 0
list_1 = []
while i < len(A_dim1):
    if A_dim1[i] == 1:
        list_1.append(i)           # list_1列表存放所有1在labels数组中的位置 存放所有正样本的位置
    i = i+1
num1 = len(list_1)                   # 正样本的数量,共1923个1
group_size1 = int(num1/fold)          # 计算每折1的个数，每折有192个1
random.seed(10)
random.shuffle(list_1)             # 将1的位置随机打乱

# 将1分为10组， 存放在grouped_data1[10*group_size]中
array_1 = np.array(list_1)[0:fold*group_size1]            # 舍弃最后的多余的正样本
grouped_data1 = np.reshape(array_1, (fold, group_size1)) #分成fold组[10,192]
np.savetxt("divide_test/index_1.txt", grouped_data1)       #保存每一折正样本的索引

#划分负样本
i = 0
list_0 = []
while i < len(A_dim1):
    if A_dim1[i] == 0:
        list_0.append(i)           # list_0列表存放所有0在labels数组中的位置
    i = i+1
num0 = len(list_0)                   # 得到0的个数  1068573
group_size0 = int(num0/fold)          # 计算每折0的个数，每折有106857个0
random.seed(10)
random.shuffle(list_0)             # 将0的位置随机打乱

grouped_data0 = [list_0[i * group_size1:(i + 1) * group_size1] for i in range(fold)]
np.savetxt("divide_test/index_0.txt", grouped_data0)

#为每折交叉验证生成特征矩阵和邻接矩阵
f = 0
while f < fold:
    DT_feature = np.zeros((m * n,  m + n))
    i = 0
    DTI=copy.deepcopy(A_WKNKN)#用WKNKN补全后的矩阵作为初始矩阵
    while i < group_size1: #将一折中的1变为0 将正样本设置成0
        r = int(grouped_data1[f, i] / n)
        c = int(grouped_data1[f, i] % n)
        DTI[r, c] = 0
        i += 1  # 得到每次交叉验证中所使用的A矩阵
    feature_matrix1 = np.hstack((SD, DTI))  #【708,2220】
    feature_matrix2 = np.hstack((DTI.T, ST))  #【1512，2220】
    X = np.vstack((feature_matrix1, feature_matrix2))  # 2220*2220
    np.savetxt("divide_test/X"+str(f)+".txt", X)

    feature_matrix3 = np.hstack((DDI, DTI))
    feature_matrix4 = np.hstack((DTI.T, TTI))
    adj = np.vstack((feature_matrix3, feature_matrix4))  # 2220*2220
    np.savetxt("divide_test/A" + str(f) + ".txt", adj)
    f += 1
print('end')
