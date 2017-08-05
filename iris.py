# -*- coding: utf-8 -*-
##第一行一定要先加 (有關中文編碼) .
##Alt+3 Alt+4 快速加註 減註
#
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# 註解
#print(iris_X[:2, :])       #取2個樣品, 敘述其型態
#print(iris_y)

X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3)

#print(y_train) 	        #打亂學習數據

knn = KNeighborsClassifier()    	            
knn.fit(X_train,y_train)                        #放入train data
print(knn.predict(X_test))                  # train 好的去預測
print(y_test)                                           # 真實的數字


