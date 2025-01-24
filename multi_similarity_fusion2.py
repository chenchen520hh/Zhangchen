import numpy as np
from tqdm import tqdm

# A可能不是方阵，A是numpy类型的数组
# 计算Jaccard相似性
def Jaccard_similarity(A):
    # B是返回的杰卡德相似性矩阵
    B=np.zeros((A.shape[0],A.shape[0]))
    for i in tqdm(range(B.shape[0])):
        for j in range(i+1,B.shape[1]):#只做上三角部分
            #下面的代码计算药物i和药物j的相似性
            if np.sum(A[i])==0 and np.sum(A[j])==0:#如果两个药物不和任何靶点相互作用，则两个药物的相似度为0
                B[i][j]=0
            else:
                jiaoji=0
                bingji=0
                # 计算A[i]和A[j]的交集和并集
                for k in range(A.shape[1]):
                    if A[i][k]==1 and A[j][k]==1:
                        jiaoji+=1
                        bingji+=1
                    elif A[i][k]==1 or A[j][k]==1:
                        bingji+=1
                B[i][j]=jiaoji/bingji
    # 此时B只是上三角矩阵，将上三角矩阵转为对称阵
    # 将主对角元素置为1
    # 因为有些药物不和任何靶点相互作用，但是自己和自己的相似度肯定是1
    row,col=np.diag_indices_from(B)
    B[row,col]=1
    B += B.T - np.diag(B.diagonal())
    return B
if __name__=="__main__":
    # 药物相似性矩阵融合
    drug_drug_interaction=np.loadtxt('dataset2/mat_data/mat_drug_drug.txt')#药物-药物相互作用矩阵[708,708]
    drug_disease_association=np.loadtxt('dataset2/mat_data/mat_drug_disease.txt')#药物疾病[708,5603]
    drug_sideeffect_association=np.loadtxt('dataset2/mat_data/mat_drug_se.txt')#药物副作用 [708,4192]
    drug_drug_chemistry_similarity=np.loadtxt('dataset2/mat_data/Similarity_Matrix_Drugs.txt') #药物化学结构[708,708]
    # 前三个矩阵需要求jaccard相似度
    drug_drug_interaction_similarity=Jaccard_similarity(drug_drug_interaction)
    drug_disease_association_similarity=Jaccard_similarity(drug_disease_association)
    drug_sideeffect_association_similarity=Jaccard_similarity(drug_sideeffect_association)
    # 保存一下把，可能计算的比较慢
    np.save('dataset2/mat_data/drug_drug_interaction_similarity.npy',drug_drug_interaction_similarity)
    np.save('dataset2/mat_data/drug_disease_association_similarity.npy',drug_disease_association_similarity)
    np.save('dataset2/mat_data/drug_sideeffect_association_similarity.npy',drug_sideeffect_association_similarity)
    np.save('dataset2/mat_data/drug_drug_chemistry_similarity.npy',drug_drug_chemistry_similarity)
    #将四个矩阵融合
    x1=np.maximum(drug_drug_chemistry_similarity,drug_drug_interaction_similarity)
    x2=np.maximum(x1, drug_disease_association_similarity)
    drug_fusion_similarity=np.maximum(x2,drug_sideeffect_association_similarity)
    np.savetxt('dataset2/mat_data/drug_fusion_similarity_2214_2214.txt',drug_fusion_similarity)


    # 靶点相似性矩阵融合
    target_disease_association=np.loadtxt('dataset2/mat_data/mat_protein_disease.txt')
    target_target_interaction=np.loadtxt('dataset2/mat_data/mat_protein_protein.txt')
    target_target_sequence_similarity=np.loadtxt('dataset2/mat_data/Similarity_Matrix_Proteins.txt')
    # 前两个矩阵需要求jaccard相似度
    target_disease_association_similarity=Jaccard_similarity(target_disease_association)
    target_target_interaction_similarity=Jaccard_similarity(target_target_interaction)
    np.save('dataset2/mat_data/target_disease_association_similarity.npy',target_disease_association_similarity)
    np.save('dataset2/mat_data/target_target_interaction_similarity.npy',target_target_interaction_similarity)
    np.save('dataset2/mat_data/target_target_sequence_similarity.npy',target_target_sequence_similarity)
    y1=np.maximum(target_target_sequence_similarity,target_disease_association_similarity)#让程序运行到96行
    target_fusion_similarity=np.maximum(y1,target_target_interaction_similarity)
    np.savetxt('dataset2/mat_data/target_fusion_similarity_1968_1968.txt',target_fusion_similarity)
    print('end')

