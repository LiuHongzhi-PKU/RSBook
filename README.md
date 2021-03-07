# 《推荐系统》教材算法实现

## 项目介绍
本项目为《推荐系统》[刘宏志著]教材中的算法实现。

## 教材简介 
本书除了介绍推荐系统的一般框架、典型应用和评测方法之外，还主要介绍各种典型推荐算法的思想、原理、算法设计和应用场景，包括针对“千人千面”的个性化推荐和针对“千人万面”的情境化推荐。此外，本书还包含一些和推荐系统相关的专题内容，如针对排序问题的排序学习和针对信息融合的异质信息网络模型。 本书可作为计算机科学与技术、软件工程、数据科学与大数据技术、人工智能等专业的高年级本科生和研究生的相关课程教材，也可作为从事推荐系统、搜索引擎、数据挖掘等研发工作相关人员的参考书。


刘宏志 编著.《推荐系统》. ISBN: 9787111649380. 机械工业出版社. 2020。    
http://www.cmpedu.com/books/book/5601946.htm  
  <img src="" width = "300" height = "400" alt="" align="教材封面" />


## 主要贡献人
王缤 周昭育 王澳博 刘鸿达    

## 开发环境
python 3.7.3            
numpy 1.16          
pytorch 1.1.0           

## 代码目录
data：算法测试用到的数据集     
Ch2：基于邻域的协同过滤    
Ch2-jupyter：基于邻域的协同过滤，jupyter代码，对应教材上的示例    
Ch3：基于模型的协同过滤    
Ch4：基于内容和知识的推荐    
Ch5：混合推荐系统    
Ch7：基于排序学习的推荐    
Ch9：基于时空信息的推荐    
Ch10：基于社交的推荐    
Ch11：基于异质信息网络的推荐    

## 算法列表
### 第二章 基于邻域的协同过滤
|算法名|对应文件|算法类型|
|----|----|----|
|针对TopN推荐的userCF|userCF_TopN.py|TopN推荐|
|针对评分预测的userCF|userCF_rating.py|评分预测|
|针对TopN推荐的itemCF|itemCF_TopN.py|TopN推荐|
|针对评分预测的itemCF|itemCF_rating.py|评分预测|
|基于距离的相似度度量|itemCF_dis.py|TopN推荐|
|slopeOne算法|SlopeOne.py|评分预测|
|激活扩散模型|spreadingActivation.py|TopN推荐|
|物质扩散模型|spreadingSubstance.py|TopN推荐|
|热传导模型|thermalConduction.py|TopN推荐|


### 第三章 基于模型的协同过滤
|算法名|对应文件|算法类型|
|----|----|----|
|Apriori|Apriori.py|
|LFM|LFM.py|评分预测|
|SVD|svd.py|
|SVD++|SVD++.py|评分预测|
|CPMF|CPMF.py|评分预测|
|WRMF|WRMF.py|TopN推荐|
|负样本欠采样|under_sampling.py|TopN推荐|

### 第四章 基于内容和知识的推荐
|算法名|对应文件|算法类型|
|----|----|----|
|基于词向量空间模型的文本表示（TF-IDF模型）|tfidfCF.py|TopN推荐|
|朴素item_cf|itemCF_news.py|TopN推荐|
|基于语料库的文本相似度（PMI-IR）|PMI_IR_CF.py|TopN推荐|
|基于约束的推荐|基于约束的推荐——MinRelax.ipynb|
|基于效用的推荐|基于效用的推荐.ipynb|
|基于实例的推荐|基于实例的推荐.ipynb|

### 第五章 混合推荐系统
|算法名|对应文件|算法类型|
|----|----|----|
|LFM1|LFM与平均融合.ipynb|评分预测|
|LFM2|LFM与平均融合.ipynb|评分预测|
|平均融合|LFM与平均融合.ipynb|评分预测|
|逻辑回归stacking|逻辑回归stacking.ipynb|评分预测|
|波达计数|波达计数.ipynb|

### 第七章 基于排序学习的推荐
|算法名|对应文件|算法类型|
|----|----|----|
|BPR|BPR.py|TopN推荐|
|CPLR|CPLR.py|TopN推荐|
|P-PushCR|P-PushCR.py|TopN推荐|
|RankBoost|RankBoost.py|TopN推荐|
|RankingSVM|RankingSVM.py|TopN推荐|
|RankNet|RankNet.py|TopN推荐|

### 第九章 基于时空信息的推荐
|算法名|对应文件|算法类型|
|----|----|----|
|最近最热门推荐算法|recentPopular.py|TopN推荐|
|基于时间的项目协同过滤|itemCF_TopN_Time.py|TopN推荐|
|基于时间的用户协同过滤|userCF_TopN_Time.py|TopN推荐|
|基于会话的协同过滤|itemCF_session.py|TopN推荐|
|基于会话的扩散|sessionSpreading.py|TopN推荐|
|基于统一转移矩阵的序列预测|Markov.py|TopN推荐|
|基于个性化转移矩阵的序列预测|FPMC.py|TopN推荐|
|基于循环神经网络的序列预测（RNN）|RNN.py|TopN推荐|
|基于循环神经网络的序列预测（LSTM）|LSTM.py|TopN推荐|
|基于循环神经网络的序列预测（GRU）|GRU.py|TopN推荐|
|基于空间信息的推荐：预过滤|LBSpre.py|TopN推荐|
|基于空间信息的推荐：后过滤|LBSpost.py|TopN推荐|
|基于空间信息的推荐：情景化建模|LBSmodel.py|TopN推荐|

### 第十章 基于社交的推荐
|算法名|对应文件|算法类型|
|----|----|----|
|RSTE|RSTE.py|评分预测|
|SoRec|SoRec.py|评分预测|
|SoReg|SoReg.py|评分预测|
|SocialMF|SocialMF.py|评分预测|
|基于社交信息的物质扩散|socialSpreadingSubstance.py|TopN推荐|
|基于社交信息的UserCF|socialUserCF.py|评分预测|

### 第十一章 基于异质信息网络的推荐 
|算法名|对应文件|算法类型|
|----|----|----|
|pathSim|pathSim_special.ipynb||
|randomWalk|randomWalk.ipynb||
|heteMF|heteMF.py|TopN推荐|
|heteCF|heteCF.py|TopN推荐|
|FMG|FMG_data_gen.ipynb、FMG.ipynb|model|
|SemRec|SemRec.py|TopN推荐|


