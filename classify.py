'''
Generate by Python3.5

'''
import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import data_utils

'''

原数据格式:
| 类别:
unacc, acc, good, vgood

| 特征:
buying:   vhigh, high, med, low.
maint:    vhigh, high, med, low.
doors:    2, 3, 4, 5more.
persons:  2, 4, more.
lug_boot: small, med, big.
safety:   low, med, high.

'''
#数据链接:
url="http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"


#进行数值化处理的dict:
str2int={
    'vhigh':4,
    'high':3,
    'med':2,
    'low':1,
    '2':2,
    '3':3,
    '4':4,
    '5more':5,
    'more':5,
    'small':3,
    'big':1,
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3,
}

X,y=data_utils.dispose_data(url,str2int)
#将数据集切分为训练集和测试集:
train_data,test_data,train_target,test_target=\
    train_test_split(X,y,test_size=0.3,random_state=0)
print("决策树开始训练!")
tree_train_start=time.time()
clf_tree=DecisionTreeClassifier(\
                    criterion='entropy', splitter='best', \
                    max_depth=None, min_samples_split=1,\
                     min_samples_leaf=1, min_weight_fraction_leaf=0.0,\
                      max_features=None, random_state=42,\
                       max_leaf_nodes=None,\
                        class_weight=None, presort=False)
clf_tree=clf_tree.fit(train_data,train_target)
tree_train_time=time.time()-tree_train_start
print("决策树训练结束!\n")

print("SVM开始训练!")
svm_train_start=time.time()
clf_svm=SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', \
                    coef0=0.0, shrinking=True, probability=False,\
                     tol=0.001, cache_size=200, class_weight=None,\
                      verbose=False, max_iter=-1, \
                      decision_function_shape=None, random_state=None)
clf_svm=clf_svm.fit(train_data,train_target)
svm_train_time=time.time()-svm_train_start
print("SVM训练结束!\n")

print("开始测试!")
score_tree=clf_tree.score(test_data,test_target)
score_svm=clf_svm.score(test_data,test_target)
print("测试结束!\n")
print("tree时间{0} 准确率{1}".format(tree_train_time*10000,score_tree))
print("svm 时间{0} 准确率{1}".format(svm_train_time*10000,score_svm))

