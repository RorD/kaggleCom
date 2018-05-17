# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:17:26 2018

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
#from sklearn.metrics import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
#import csv
from sklearn.decomposition import PCA

# 加载数据
def opencsv():
    # 使用 pandas 打开
    data = pd.read_csv('F:/Project/Kaggle/digitalRecognition/digit-recognizer/train.csv')
    data1 = pd.read_csv('F:/Project/Kaggle/digitalRecognition/digit-recognizer/test.csv')

    train_data = data.values[0:, 1:]  # 读入全部训练数据
    train_label = data.values[0:, 0]
    test_data = data1.values[0:, 0:]  # 测试全部测试个数据
    return train_data, train_label, test_data

trainData, trainLabel, testData = opencsv()

# 模型训练
def knnClassify(trainData, trainLabel):
    pca = PCA(n_components=5)
    pcaFit = pca.fit(trainData)
    trainDataPca = pcaFit.transform(trainData)
    knnClf = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto')   # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainDataPca, np.ravel(trainLabel))
    return pcaFit,knnClf

pcaFit,knnClf = knnClassify(trainData, trainLabel)

# 结果预测
testDataPca = pcaFit.transform(testData)
testLabel = knnClf.predict(testDataPca)




writeLabel = []
index=0
for i in range(len(testLabel)):
    line = []
    index = index+1
    line.append(index)
    line.append(testLabel[i])
    writeLabel.append(line)

columns_=['ImageId', 'Label']
labelDf = pd.DataFrame(writeLabel,columns=columns_)
labelDf.to_csv('F:/Project/Kaggle/digitalRecognition/digit-recognizer/Result_sklearn_knn.csv')


#def saveResult(result, csvName):
#    with open(csvName, 'wb') as myFile:
#        myWriter = csv.writer(myFile)
#        #myWriter.writerow(["ImageId", "Label"])
#        index = 0
#        for i in result:
#            tmp = []
#            index = index+1
#            tmp.append(index)
#            # tmp.append(i)
#            tmp.append(int(i))
#            myWriter.writerow(tmp)
# 结果的输出
#saveResult(testLabel, 'F:/Project/Kaggle/digitalRecognition/digit-recognizer/Result_sklearn_knn.csv')










