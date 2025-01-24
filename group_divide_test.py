import random
import copy
import numpy as np

fold = 10
m = 2214  # 药物数量
n = 1968  # 靶标数量

# 读取数据
SD = np.loadtxt('dataset2/mat_data/drug_fusion_similarity_2214_2214.txt')  # 药物相似度矩阵
ST = np.loadtxt('dataset2/mat_data/target_fusion_similarity_1968_1968.txt')  # 靶标相似度矩阵
DDI = np.loadtxt("dataset2/mat_data/mat_drug_drug.txt")  # 药物相互作用
TTI = np.loadtxt("dataset2/mat_data/mat_protein_protein.txt")  # 靶标相互作用
A = np.loadtxt("dataset2/mat_data/mat_drug_protein.txt")  # A矩阵 原始标签数据
A_WKNKN = np.loadtxt('dataset2/sim_network/DTI_2214_1968_WKNKN_MAX_DISCRETIZE.txt')  # 经过WKNKN补全后的邻接矩阵

A_dim1 = A.flatten()  # 展平矩阵
list_1 = [i for i in range(len(A_dim1)) if A_dim1[i] == 1]  # 存放正样本的位置
num1 = len(list_1)  # 正样本的数量

group_size1 = int(num1 / fold)  # 每折正样本数量
random.seed(10)
random.shuffle(list_1)  # 随机打乱正样本位置

# 舍弃最后的多余的正样本，分成fold组
array_1 = np.array(list_1)[:fold * group_size1]
grouped_data1 = np.reshape(array_1, (fold, group_size1))
np.savetxt("divide_test1/index_1.txt", grouped_data1)

# 负样本
list_0 = [i for i in range(len(A_dim1)) if A_dim1[i] == 0]  # 存放负样本的位置
random.seed(10)
random.shuffle(list_0)  # 随机打乱负样本位置

# 确保每折的正负样本数量相同
grouped_data0 = [list_0[i * group_size1:(i + 1) * group_size1] for i in range(fold)]
np.savetxt("divide_test1/index_0.txt", grouped_data0)

# 为每折交叉验证生成特征矩阵和邻接矩阵
for f in range(fold):
    # 使用WKNKN补全后的矩阵作为初始矩阵
    DTI = copy.deepcopy(A_WKNKN)

    # 将当前折的正样本和负样本设置为0（即这些位置将作为测试集，不参与训练）
    for idx in grouped_data1[f]:  # 正样本设置为0
        r = idx // n
        c = idx % n
        DTI[r, c] = 0

    for idx in grouped_data0[f]:  # 负样本设置为0
        r = idx // n
        c = idx % n
        DTI[r, c] = 0

    # 构建特征矩阵X
    feature_matrix1 = np.hstack((SD, DTI))  # [2214, 4182]
    feature_matrix2 = np.hstack((DTI.T, ST))  # [1968, 4182]
    X = np.vstack((feature_matrix1, feature_matrix2))  # [4182, 4182]
    np.savetxt(f"divide_test1/X_{f}.txt", X)

    # 保存邻接矩阵
    feature_matrix3 = np.hstack((DDI, DTI))
    feature_matrix4 = np.hstack((DTI.T, TTI))
    adj = np.vstack((feature_matrix3, feature_matrix4))
    np.savetxt(f"divide_test1/A_{f}.txt", adj)

print('end')