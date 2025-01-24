import pandas as pd
import matplotlib.pyplot as plt

#plt.use('TkAgg')

# 读取保存的Excel文件
fold = 1  # 根据你的情况选择fold编号
file_path = 'results_fold_' + str(fold) + '.xlsx'
file_path1 = 'results_fold____' + str(fold) + '.xlsx'
file_path2 = 'results_fold__' + str(fold) + '.xlsx'
file_path3 = 'results_fold_ptGCN_' + str(fold) + '.xlsx'
file_path4 = 'results_fold___' + str(fold) + '.xlsx'


# 读取AUC和AUPR的结果
df_auc_aupr = pd.read_excel(file_path, sheet_name='AUC and AUPR')
AUC = df_auc_aupr['AUC'].values[0]
AUPR = df_auc_aupr['AUPR'].values[0]

df_auc_aupr1 = pd.read_excel(file_path1, sheet_name='AUC and AUPR')
AUC1 = df_auc_aupr1['AUC'].values[0]
AUPR1 = df_auc_aupr1['AUPR'].values[0]

df_auc_aupr2 = pd.read_excel(file_path2, sheet_name='AUC and AUPR')
AUC2 = df_auc_aupr2['AUC'].values[0]
AUPR2 = df_auc_aupr2['AUPR'].values[0]

df_auc_aupr3 = pd.read_excel(file_path3, sheet_name='AUC and AUPR')
AUC3 = df_auc_aupr3['AUC'].values[0]
AUPR3 = df_auc_aupr3['AUPR'].values[0]

df_auc_aupr4 = pd.read_excel(file_path4, sheet_name='AUC and AUPR')
AUC4= df_auc_aupr4['AUC'].values[0]
AUPR4 = df_auc_aupr4['AUPR'].values[0]
# 读取ROC和PR曲线的数据
df_roc = pd.read_excel(file_path, sheet_name='ROC Curve')
df_pr = pd.read_excel(file_path, sheet_name='PR Curve')

df_roc1 = pd.read_excel(file_path1, sheet_name='ROC Curve')
df_pr1 = pd.read_excel(file_path1, sheet_name='PR Curve')

df_roc2 = pd.read_excel(file_path2, sheet_name='ROC Curve')
df_pr2 = pd.read_excel(file_path2, sheet_name='PR Curve')

df_roc3 = pd.read_excel(file_path3, sheet_name='ROC Curve')
df_pr3 = pd.read_excel(file_path3, sheet_name='PR Curve')

df_roc4 = pd.read_excel(file_path4, sheet_name='ROC Curve')
df_pr4= pd.read_excel(file_path4, sheet_name='PR Curve')
# 将DataFrame列转换为numpy数组
fpr = df_roc['FPR'].values  # 将DataFrame列转为numpy数组
tpr = df_roc['TPR'].values  # 同样处理TPR列

fpr1 = df_roc1['FPR'].values  # 将DataFrame列转为numpy数组
tpr1 = df_roc1['TPR'].values  # 同样处理TPR列

fpr2 = df_roc2['FPR'].values  # 将DataFrame列转为numpy数组
tpr2 = df_roc2['TPR'].values  # 同样处理TPR列

fpr3 = df_roc3['FPR'].values  # 将DataFrame列转为numpy数组
tpr3 = df_roc3['TPR'].values  # 同样处理TPR列

fpr4 = df_roc4['FPR'].values  # 将DataFrame列转为numpy数组
tpr4 = df_roc4['TPR'].values  # 同样处理TPR列

precision = df_pr['Precision'].values  # 将PR曲线的Precision列转为numpy数组
recall = df_pr['Recall'].values  # 同样处理Recall列

precision1 = df_pr1['Precision'].values  # 将PR曲线的Precision列转为numpy数组
recall1 = df_pr1['Recall'].values  # 同样处理Recall列

precision2 = df_pr2['Precision'].values  # 将PR曲线的Precision列转为numpy数组
recall2 = df_pr2['Recall'].values  # 同样处理Recall列

precision3 = df_pr3['Precision'].values  # 将PR曲线的Precision列转为numpy数组
recall3 = df_pr3['Recall'].values  # 同样处理Recall列

precision4 = df_pr4['Precision'].values  # 将PR曲线的Precision列转为numpy数组
recall4 = df_pr4['Recall'].values  # 同样处理Recall列

# 画ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='blue',label='VHGAE (AUROC = %0.4f)' % AUC)
plt.plot(fpr1, tpr1, color='yellow',label='Without_Fusion (AUROC = %0.4f)' % AUC1)
plt.plot(fpr2, tpr2, color='red',label='Without_EM (AUROC = %0.4f)' % AUC2)
plt.plot(fpr3, tpr3, color='orange',label='Without_WGCN (AUROC = %0.4f)' % AUC3)
plt.plot(fpr4, tpr4, color='green',label='Without_WKNN (AUROC = %0.4f)' % AUC4)
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
('10curves/' + str(fold) + '_ROC.jpg')
plt.show()

# 画PR曲线
plt.figure()
plt.plot(recall, precision,color='blue', label='VHGAE (AUPR = %0.4f)' % AUPR)
plt.plot(recall1, precision1,color='yellow', label='Without_Fusion (AUPR = %0.4f)' % AUPR1)
plt.plot(recall2, precision2, color='red',label='Without_EM (AUROC = %0.4f)' % AUPR2)
plt.plot(recall3, precision3,color='orange', label='Without_WGCN (AUPR = %0.4f)' % AUPR3)
plt.plot(recall4, precision4,color='green', label='Without_WKNN (AUPR = %0.4f)' % AUPR4)
plt.title('PR curve')
plt.xlabel("RECALL")
plt.ylabel("PRECISION")
plt.legend(loc="lower right")
plt.savefig('10curves/' + str(fold) + '_PR.jpg')
plt.show()

print('end')