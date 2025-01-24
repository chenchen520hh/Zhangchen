import numpy as np
import math
import copy
#致密化DTI矩阵 使用加权k最近邻算法
def WKNKN(DTI,drugSimilarity,proteinSimilarity,K,r):
    drugCount=DTI.shape[0]
    proteinCount=DTI.shape[1]
    # 标志drug是new drug还是known drug 如果是known drug,则相应位为1 如果是new drug,则相应位为0
    flagDrug=np.zeros([drugCount])
    flagProtein=np.zeros([proteinCount])
    for i in range(drugCount):
        for j in range(proteinCount):
            if(DTI[i][j]==1):
                flagDrug[i]=1
                flagProtein[j]=1
    #定义两个[708,1512]的矩阵
    Yd=np.zeros([drugCount,proteinCount])
    Yt=np.zeros([drugCount,proteinCount])
    # Yd矩阵的获取
    for d in range(drugCount):
        dnn=KNearestKnownNeighbors(d,drugSimilarity,K,flagDrug)#返回K近邻的下标
        w=np.zeros([K])
        Zd=0
        # 获取权重w和归一化因子Zd
        for i in range(K):
            w[i]=math.pow(r,i)*drugSimilarity[d,dnn[i]] #每个邻接节点的权重
            Zd+=drugSimilarity[d,dnn[i]] #加权求和
        for i in range(K):
            Yd[d]=Yd[d]+w[i]*DTI[dnn[i]]
        Yd[d]=Yd[d]/Zd #获得每个药物节点的相似度
    # Yt矩阵的获取
    for t in range(proteinCount):
        tnn=KNearestKnownNeighbors(t,proteinSimilarity,K,flagProtein)
        w=np.zeros([K])
        Zt=0
        for j in range(K):
            w[j]=math.pow(r,j)*proteinSimilarity[t,tnn[j]]
            Zt+=proteinSimilarity[t,tnn[j]]
        for j in range(K):
            Yt[:,t]=Yt[:,t]+w[j]*DTI[:,tnn[j]]
        Yt[:,t]=Yt[:,t]/Zt
    Ydt=Yd+Yt
    Ydt=Ydt/2
    ans=np.maximum(DTI,Ydt)#ans的形状是[drugCount,proteinCount] 选DTI和Ydt最大值
    return ans
# 返回下标，node结点的K近邻（不包括new drug/new target） 用于找到给定节点的k个最近邻
def KNearestKnownNeighbors(node,matrix,K,flagNodeArray):
    KknownNeighbors=np.array([])
    featureSimilarity=matrix[node].copy()#在相似性矩阵中取出第node行
    featureSimilarity[node]=-100   #排除自身结点,使相似度为-100
    featureSimilarity[flagNodeArray==0]=-100  #排除new drug/new target,使其相似度为-100 只考虑已知药物不考虑未知药物，因为未知药物的相互作用都是0
    # 只考虑known node
    KknownNeighbors=featureSimilarity.argsort()[::-1]#按照相似度降序排序
    KknownNeighbors=KknownNeighbors[:K]#返回前K个结点的下标
    return KknownNeighbors
if __name__ == "__main__":
    DTI = np.loadtxt('dataset2/mat_data/mat_drug_protein.txt')
    Sd=np.loadtxt('dataset2/sim_network/Sim_mat_drugs.txt')
    St=np.loadtxt('dataset2/sim_network/Sim_mat_proteins.txt')
    predict_Y=WKNKN(DTI=DTI,drugSimilarity=Sd,proteinSimilarity=St,K=10,r=0.8) #选择10个邻居节点
    # 统计原始数据集非0的个数
    num1=0
    for i in range(DTI.shape[0]):
        for j in range(DTI.shape[1]):
            if DTI[i][j]==1:
                num1+=1
    print(num1)
    frequent_no_zero=num1/(DTI.shape[0]*DTI.shape[1])  #原始数据集非零元素比例0.2
    print("Original data none zero ratio:%.4f"%frequent_no_zero)
    #计算补全后非零元素比例
    num_float=0
    for i in range(predict_Y.shape[0]):
        for j in range(predict_Y.shape[1]):
            if predict_Y[i][j]!=0:
                num_float+=1
    print(num_float)
    frequent_no_zero=num_float/(predict_Y.shape[0]*predict_Y.shape[1]) #补全后非零元素比例3.67
    print("After WKNKN,none zero ratio:%.4f"%frequent_no_zero)
    # 离散化WKNKN  将不确定的可能性值转换为确定的相互作用关系
    num_greaterzero_samllerone=0
    for i in range(predict_Y.shape[0]):
        for j in range(predict_Y.shape[1]):
            if predict_Y[i][j]!=0 and predict_Y[i][j]!=1:
                num_greaterzero_samllerone += 1

    print('Before discretize: num of float num %d'%num_greaterzero_samllerone)

    float_array=copy.deepcopy(predict_Y[ (predict_Y>0) & (predict_Y<1) ])
    float_median=np.median(float_array) #找中间值作为阈值
    predict_Y[predict_Y>=float_median]=1
    predict_Y[predict_Y<float_median]=0
    num_greaterzero_samllerone=0
    for i in range(predict_Y.shape[0]):
        for j in range(predict_Y.shape[1]):
            if predict_Y[i][j]!=0 and predict_Y[i][j]!=1:
                num_greaterzero_samllerone += 1
    print('After discretize: num of float %d'%num_greaterzero_samllerone)
#    np.savetxt('dataset2/sim_network/DTI_2214_1968_WKNKN_MAX_DISCRETIZE.txt',predict_Y)
    print('end!!')




