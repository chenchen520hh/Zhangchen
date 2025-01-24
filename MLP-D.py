import pandas as pd
import matplotlib.pyplot as plt

#plt.use('TkAgg')

# 读取保存的Excel文件
fold = 1  # 根据你的情况选择fold编号
file_path = 'results_fold_' + str(fold) + '.xlsx'
file_path1 = 'results_fold_Decoder_' + str(fold) + '.xlsx'



# 读取AUC和AUPR的结果
df_auc_aupr = pd.read_excel(file_path, sheet_name='AUC and AUPR')
AUC = df_auc_aupr['AUC'].values[0]
AUPR = df_auc_aupr['AUPR'].values[0]

df_auc_aupr1 = pd.read_excel(file_path1, sheet_name='AUC and AUPR')
AUC1 = df_auc_aupr1['AUC'].values[0]
AUPR1 = df_auc_aupr1['AUPR'].values[0]

# 读取ROC和PR曲线的数据
df_roc = pd.read_excel(file_path, sheet_name='ROC Curve')
df_pr = pd.read_excel(file_path, sheet_name='PR Curve')

df_roc1 = pd.read_excel(file_path1, sheet_name='ROC Curve')
df_pr1 = pd.read_excel(file_path1, sheet_name='PR Curve')

fpr = df_roc['FPR'].values  # 将DataFrame列转为numpy数组
tpr = df_roc['TPR'].values  # 同样处理TPR列

fpr1 = df_roc1['FPR'].values  # 将DataFrame列转为numpy数组
tpr1 = df_roc1['TPR'].values  # 同样处理TPR列


precision = df_pr['Precision'].values  # 将PR曲线的Precision列转为numpy数组
recall = df_pr['Recall'].values  # 同样处理Recall列

precision1 = df_pr1['Precision'].values  # 将PR曲线的Precision列转为numpy数组
recall1 = df_pr1['Recall'].values  # 同样处理Recall列


# 画ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='blue',label='InnerProductDecoder (AUROC = %0.4f)' % AUC)
plt.plot(fpr1, tpr1, color='red',label='BilinearDecoder (AUROC = %0.4f)' % AUC1)

plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
('10curves/' + str(fold) + '_ROC.jpg')
plt.show()

# 画PR曲线
plt.figure()
plt.plot(recall, precision,color='blue', label='InnerProductDecoder (AUPR = %0.4f)' % AUPR)
plt.plot(recall1, precision1,color='red', label='BilinearDecoder (AUPR = %0.4f)' % AUPR1)
plt.title('PR curve')
plt.xlabel("RECALL")
plt.ylabel("PRECISION")
plt.legend(loc="lower right")
plt.savefig('10curves/' + str(fold) + '_PR.jpg')
plt.show()

print('end')