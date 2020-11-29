# lec10

第十章全部算法使用的数据集均为精简的Epinions。  


## 各算法对应文件
算法名	                    对应文件  
RSTE	                    RSTE.py  
SoRec	                    SoRec.py  
SoReg	                    SoReg.py  
SocialMF	                SocialMF.py  
基于社交信息的物质扩散	      socialSpreadingSubstance.py  
基于社交信息的UserCF	     socialUserCF.py  




## TopN推荐算法运行结果
算法名	                Precision	Recall	Coverage  
基于社交信息的物质扩散	    0.016	0.042	0.039  


## 评分预测算法运行结果
算法名	                MAE	        RMSE  
RSTE	                0.208	0.281  
SoRec	                0.334	0.432  
SoReg	                0.208	0.281  
SocialMF	            0.196	0.271  
基于社交信息的UserCF	  0.190	0.251  


注：数据集中的评分信息已被归一化至0~1之间。  






