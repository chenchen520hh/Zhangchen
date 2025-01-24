from my_function import *
import numpy as np
from pylab import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve,roc_auc_score,average_precision_score
from tqdm import tqdm
# FOLD = 10
from matplotlib import pyplot as plt
#计算和绘制第 fold 折的药物-靶标相互作用（DTI）的预测结果的ROC曲线和PR曲线，并计算AUC和AUPR值。 （AUC是ROC曲线下的面积，
fold = 0 #查看第fold折的测试集结果
DTI = np.loadtxt("whole_data/mat_drug_protein.txt")#标签矩阵
index1 = np.loadtxt('divide_1/index_1.txt')
index0 = np.loadtxt('divide_1/index_0.txt')
index = np.hstack((index1, index0))
drug_num = DTI.shape[0]#708个药物
protein_num = DTI.shape[1]#1512个靶点
score = np.zeros(DTI.shape)
mask=np.zeros(DTI.shape)#用于标识是否是该折的数据
#读取预测结果
pre = np.loadtxt('predict_result_of_VGAE/VGAE'+str(fold)+'.txt')
idx = index[fold, :]
# pre = line_normalize(pre)
# pre = pre[:, 1]
#填充得分矩阵和掩码矩阵
for i in range(len(idx)):
    d = int(idx[i] / protein_num)
    p = int(idx[i] % protein_num)
    score[d, p] += pre[i]
    mask[d , p] = 1

DTI=DTI.tolist()
score=score.tolist()
# mask=mask.tolist()
#计算每个药物的ROC和PR曲线
auc_list = []
aupr_list = []
tpr_list = []
fpr_list = []
recall_list = []
precision_list = []
c=0
for i in tqdm(range(drug_num)):#针对每一个药物而言
    if np.sum(np.array(DTI[i])[mask[i]==1]) == 0:
        c += 1
        continue
    else:
        tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(DTI[i])[mask[i]==1], np.array(score[i])[mask[i]==1])
        # tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(DTI[i]), np.array(score[i]))
        if len(fpr1) < 2 or len(tpr1) < 2:
            print(f"Warning: Skipping drug {i} due to insufficient points for AUC calculation.")
            continue
        fpr_list.append(fpr1)
        tpr_list.append(tpr1)
        precision_list.append(precision1)
        recall_list.append(recall1)
        auc_list.append(auc(fpr1, tpr1))
        aupr_list.append(auc(recall1, precision1)+recall1[0]*precision1[0])
#计算平均的TPR, FPR, Precision, Recall
tpr = equal_len_list(tpr_list)
fpr = equal_len_list(fpr_list)
precision = equal_len_list(precision_list)
recall = equal_len_list(recall_list)
tpr=np.array(tpr)
fpr=np.array(fpr)
precision=np.array(precision)
recall=np.array(recall)

tpr_mean = np.mean(tpr, axis=0)
fpr_mean = np.mean(fpr, axis=0)
recall_mean = np.mean(recall, axis=0)
precision_mean = np.mean(precision, axis=0)

#计算AUC和AUPR
AUC=auc(fpr_mean, tpr_mean)
AUPR=auc(recall_mean, precision_mean)+recall_mean[0]*precision_mean[0] #第(recall_mean[0],precision_mean[0])点的P值最高，R值最低，也就是PR曲线最左端的点
# np.savetxt('10curves/tpr_mean_'+str(fold)+'.txt',tpr_mean)
# np.savetxt('10curves/fpr_mean_'+str(fold)+'.txt',fpr_mean)
# np.savetxt('10curves/recall_mean_'+str(fold)+'.txt',recall_mean)
# np.savetxt('10curves/precision_mean_'+str(fold)+'.txt',precision_mean)

print('The auc of prediction is:%.4f'%AUC)
print('The aupr of prediction is:%.4f'%AUPR)
# 画ROC曲线
plt.figure()
plt.plot(fpr_mean,tpr_mean,label='ROC(AUC = %0.4f)' % AUC)

plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.savefig('10curves/'+str(fold)+'_ROC.jpg')
plt.show()
# 画PR曲线
plt.figure()
plt.plot(recall_mean,precision_mean,label='PR(AUPR = %0.4f)' % AUPR)
plt.title('PR curve')
plt.xlabel("RECALL")
plt.ylabel("PRECISION")
plt.legend(loc="lower right")
plt.savefig('10curves/'+str(fold)+'_PR.jpg')
plt.show()
print('end')















