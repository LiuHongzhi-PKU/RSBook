# lec9

获得参数说明，使用python XXX.py --help  
如python userCF_rating.py –help  



## 各算法对应文件及数据集
算法名称	                        对应文件	                数据集  
最近最热门推荐算法	                recentPopular.py	        movie-lens 100k  
基于时间的项目协同过滤	            itemCF_TopN_Time.py	        movie-lens 100k  
基于时间的用户协同过滤	            userCF_TopN_Time.py	        movie-lens 100k  
基于会话的协同过滤	                itemCF_session.py	        movie-lens 100k  
基于会话的扩散	                    sessionSpreading.py	        movie-lens 100k  
基于统一转移矩阵的序列预测	         Markov.py	                data/trans.txt  
基于个性化转移矩阵的序列预测	    FPMC.py	                    data/trans.txt  
基于循环神经网络的序列预测（RNN）	 RNN.py	                    data/sample  
基于循环神经网络的序列预测（LSTM）	LSTM.py	                    data/sample  
基于循环神经网络的序列预测（GRU）	GRU.py	                    data/sample  
基于空间信息的推荐：预过滤	        LBSpre.py	                data/poidata/Foursquare/mydata.txt  
基于空间信息的推荐：后过滤	        LBSpost.py	                data/poidata/Foursquare/mydata.txt  
基于空间信息的推荐：情景化建模	    LBSmodel.py	                data/poidata/Foursquare/mydata.txt  



## TopN推荐算法运行结果
### 下表对应数据集为movie-lens 100k
算法名	                Precision	Recall	    Coverage  
最近最热门推荐算法	    0.1567	    0.0489	    0.0434  
基于时间的项目协同过滤	0.1700	    0.0530	    0.0933  
基于时间的用户协同过滤	0.1856  	0.0579	    0.1546  
基于会话的协同过滤	    0.08419936	0.09178130	0.05410226  
基于会话的扩散      	0.06511135	0.07097445	0.05410226  

### 下表对应数据集为data/trans.txt
算法名	                        Precision	Recall	Coverage  
基于统一转移矩阵的序列预测	        0.2830	0.6011	0.1178  
基于个性化转移矩阵的序列预测	    0.2654	0.5638	0.2407  


### 下表对应数据为data/sample
算法名	                            Recall	            MMR  
基于循环神经网络的序列预测（RNN）	65.65656565656566%	41.3641641582818%  
基于循环神经网络的序列预测（LSTM）	69.6969696969697%	50.16093412378242%  
基于循环神经网络的序列预测（GRU）	61.61616161616161%	34.969667389453484%  

### 下表对应数据为data/poidata/Foursquare/mydata.txt
算法名	                        Precision	Recall	Coverage  
基于空间信息的推荐：预过滤	        0.0082	0.0431	0.8095  
基于空间信息的推荐：后过滤	        0.0074	0.0388	0.8071  
基于空间信息的推荐：情景化建模	    0.0082	0.0431	0.8048  




