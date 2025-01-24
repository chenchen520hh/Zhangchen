from my_function import *
import numpy as np
from pylab import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve,roc_auc_score,average_precision_score
from tqdm import tqdm
# FOLD = 10
from matplotlib import pyplot as plt
import torch
#读取字典的函数
def read_dict(path):
    dicFile = open(path, 'r')  # 打开数据
    txtDict = {}  # 建立字典
    while True:
        line = dicFile.readline().strip('\n')#去除前后的换行符
        if line == '':
            break
        index = line.find(':')  # 以tab键为分割
        key = line[:index]
        value = line[index+1:]
        txtDict[key] = value  # 加入字典
    dicFile.close()
    return txtDict
#读取药靶相互作用标签 读取正负样本索引并合并
DTI = np.loadtxt("whole_data/mat_drug_protein.txt")#标签矩阵
index1 = np.loadtxt('divide_test/index_1.txt')
index0 = np.loadtxt('divide_test/index_0.txt')
index = np.hstack((index1, index0))
drug_num = DTI.shape[0]#708个药物
protein_num = DTI.shape[1]#1512个靶点
#初始化一个评分矩阵
score = np.zeros(DTI.shape)
#循环读取每一折的预测结果 ，将每折的预测分数累加到评分矩阵中。
for f in tqdm(range(10)):
    pre = np.loadtxt('CL-result-of-VGAE/VGAE'+str(0)+'.txt')
    idx = index[f, :]
    for i in range(len(idx)):
        d = int(idx[i] / protein_num) #药物索引
        p = int(idx[i] % protein_num) #靶标索引
        score[d, p] += pre[i]
# 设置TOPK的值，读取药物和靶点的ID，转换评分矩阵为张量，并获取每个药物得分最高的TOPK个靶点及其得分
TOPK=30
drug_id=np.loadtxt('whole_data/drug.txt',dtype=str)
protein_id=np.loadtxt('whole_data/protein.txt',dtype=str)
score_tensor=torch.tensor(score)
candidate_target_score,candidate_target_index=score_tensor.topk(k=TOPK,dim=1)
candidate_target_score=np.array(candidate_target_score)
candidate_target_index=np.array(candidate_target_index)

#读取药物和靶标的字典文件，将结果保存为CSV文件
ans=np.empty(shape=(0,6))
drug_dict_path='whole_data/other_information_to_be_used/drug_dict_map.txt'
target_dict_path='whole_data/other_information_to_be_used/protein_dict_map.txt'
drug_dict=read_dict(drug_dict_path)
target_dict=read_dict(target_dict_path)

for i in range(drug_num):
    drug_id_drugbank=drug_id[i]
    first_column=np.array([drug_id_drugbank]*TOPK)
    second_column=np.arange(start=1,stop=TOPK+1,step=1)
    third_column=protein_id[candidate_target_index[i]]
    forth_column=candidate_target_score[i]
    fifth_column=np.array([drug_dict[drug_id_drugbank]]*TOPK)
    sixth_column=np.array(list(map(target_dict.get,third_column)))#根据字典映射成蛋白名
    # 扩展维度，方便拼接
    first_column=np.expand_dims(first_column,axis=1)
    second_column=np.expand_dims(second_column,axis=1)
    third_column=np.expand_dims(third_column,axis=1)
    forth_column=np.expand_dims(forth_column,axis=1)
    fifth_column=np.expand_dims(fifth_column,axis=1)
    sixth_column=np.expand_dims(sixth_column,axis=1)
    drug_candidate_target=np.concatenate((first_column,second_column,third_column,forth_column,fifth_column,sixth_column),axis=1)
    ans=np.concatenate((ans,drug_candidate_target),axis=0)
#对于每个药物 获取topk预测靶标和分数，构造结果数组
np.savetxt("predict_candidate_target_include_name12.19.csv",ans,fmt="%s",delimiter=",")
print('end')

# def read_dict(path):
#     """读取字典文件"""
#     with open(path, 'r') as dicFile:
#         txtDict = {}
#         for line in dicFile:
#             line = line.strip('\n')  # 去掉换行符
#             if line == '':
#                 continue
#             index = line.find(':')  # 找到分隔符位置
#             key = line[:index]
#             value = line[index + 1:]
#             txtDict[key] = value
#     return txtDict
#
#
# # 读取药靶相互作用标签和负样本索引
# DTI = np.loadtxt("whole_data/mat_drug_protein.txt")  # 标签矩阵
# index0 = np.loadtxt('divide_test/index_0.txt')  # 负样本索引
# drug_num = DTI.shape[0]  # 药物数量
# protein_num = DTI.shape[1]  # 靶标数量
#
# # 读取药物和靶标的ID及名称映射
# drug_id = np.loadtxt('whole_data/drug.txt', dtype=str)  # 药物ID列表
# protein_id = np.loadtxt('whole_data/protein.txt', dtype=str)  # 靶标ID列表
# drug_dict = read_dict('whole_data/other_information_to_be_used/drug_dict_map.txt')  # 药物ID到名称的映射
# target_dict = read_dict('whole_data/other_information_to_be_used/protein_dict_map.txt')  # 靶标ID到名称的映射
#
# # 初始化评分矩阵
# score = np.zeros(DTI.shape)
#
# # 循环读取每一折预测结果，将每折负样本分数累加到评分矩阵中
# for f in tqdm(range(10)):
#     pre = np.loadtxt(f'CL-result-of-VGAE/VGAE{f}.txt')  # 当前折预测分数
#     idx = index0[f, :]  # 当前折的负样本索引
#     for i in range(len(idx)):
#         d = int(idx[i] / protein_num)  # 药物索引
#         p = int(idx[i] % protein_num)  # 靶标索引
#         score[d, p] += pre[i]
#
# # 构造负样本结果数组
# ans = np.empty(shape=(0, 6))
#
# for d in range(drug_num):
#     # 获取负样本中当前药物对应的靶标分数
#     target_indices = np.where(score[d] > 0)[0]
#     target_scores = score[d, target_indices]
#     sorted_indices = np.argsort(-target_scores)  # 按分数从高到低排序
#     topk_indices = sorted_indices[:10]  # 获取Top-K靶标
#
#     # 构造结果
#     for rank, idx in enumerate(topk_indices, start=1):
#         protein_idx = target_indices[idx]
#         protein_id_val = protein_id[protein_idx]
#         drug_id_val = drug_id[d]
#         score_val = target_scores[idx]
#         drug_name = drug_dict.get(drug_id_val, "Unknown")
#         protein_name = target_dict.get(protein_id_val, "Unknown")
#
#         # 将结果扩展到最终数组
#         row = np.array([drug_id_val, rank, protein_id_val, score_val, drug_name, protein_name], dtype=object)
#         ans = np.vstack([ans, row])
#
# # 保存结果为CSV文件
# np.savetxt("negative_sample_topk_predictions.csv", ans, fmt="%s", delimiter=",",
#            header="Drug_ID,Rank,Protein_ID,Score,Drug_Name,Protein_Name", comments="")
# print('Negative sample Top-K predictions saved.')


#对每个药物计算ROC和PR曲线的各项指标
# DTI=DTI.tolist()
# score=score.tolist()
# auc_list = []
# aupr_list = []
# tpr_list = []
# fpr_list = []
# recall_list = []
# precision_list = []
# c=0
# for i in tqdm(range(drug_num)):#针对每一个药物而言
#     if np.sum(np.array(DTI[i])) == 0:
#         c += 1
#         continue
#     else:
#         tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(DTI[i]), np.array(score[i]))
#         fpr_list.append(fpr1)
#         tpr_list.append(tpr1)
#         precision_list.append(precision1)
#         recall_list.append(recall1)
#         auc_list.append(auc(fpr1, tpr1))
#         aupr_list.append(auc(recall1, precision1)+recall1[0]*precision1[0])
#
# tpr = equal_len_list(tpr_list)
# fpr = equal_len_list(fpr_list)
# precision = equal_len_list(precision_list)
# recall = equal_len_list(recall_list)
# tpr=np.array(tpr)
# fpr=np.array(fpr)
# precision=np.array(precision)
# recall=np.array(recall)
#
# tpr_mean = np.mean(tpr, axis=0)
# fpr_mean = np.mean(fpr, axis=0)
# recall_mean = np.mean(recall, axis=0)
# precision_mean = np.mean(precision, axis=0)
# AUC=auc(fpr_mean, tpr_mean)
# AUPR=auc(recall_mean, precision_mean)+recall_mean[0]*precision_mean[0] #第(recall_mean[0],precision_mean[0])点的P值最高，R值最低，也就是PR曲线最左端的点
# print('The auc of prediction is:%.4f'%AUC)
# print('The aupr of prediction is:%.4f'%AUPR)
# # 画ROC曲线
# plt.figure()
# plt.plot(fpr_mean,tpr_mean,label='ROC(AUC = %0.4f)' % AUC)
# plt.title('ROC curve')
# plt.xlabel("FPR")
# plt.ylabel("TPR")
# plt.legend(loc="lower right")
# plt.savefig('final_result_evaluation/ROC.jpg')
# plt.show()
# # 画PR曲线
# plt.figure()
# plt.plot(recall_mean,precision_mean,label='PR(AUPR = %0.4f)' % AUPR)
# plt.title('PR curve')
# plt.xlabel("RECALL")
# plt.ylabel("PRECISION")
# plt.legend(loc="lower right")
# plt.savefig('final_result_evaluation/PR.jpg')
# plt.show()
#
# with open('final_result_evaluation/metrics.txt', 'w') as f:
#     print('AUC:%.6f '%AUC,  'AUPR: %.6f'%AUPR,  file=f)
#
# print('end')
#
#
# #计算整体的AUC和AUPR
# y_true = DTI.flatten()
# y_score = score.flatten()
# # 预测AUC
# fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
# AUC_ROC = roc_auc_score(y_true, y_score)
# plt.figure()
# plt.plot(fpr,tpr,label='ROC(AUC = %0.4f)' % AUC_ROC)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.title('ROC curve')
# plt.xlabel("FPR")
# plt.ylabel("TPR")
# plt.legend(loc="lower right")
# plt.savefig('final_result/ROC.jpg')
# plt.show()
# print('AUC:%.6f'%AUC_ROC)
#
# # 预测AUPR
# precision, recall, threshold = precision_recall_curve(y_true, y_score, pos_label=1)
# AUPR=average_precision_score(y_true,y_score)
# plt.figure()
# plt.plot(recall,precision,label='PR(AUPR = %0.4f)'% AUPR)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.title('PR curve')
# plt.xlabel("RECALL")
# plt.ylabel("PRECISION")
# plt.legend(loc="lower right")
# plt.savefig('final_result/PR.jpg')
# plt.show()
# print('AUPR:%.6f'%AUPR)