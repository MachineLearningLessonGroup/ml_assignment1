#-*- coding:utf-8 -*-
import random
from sklearn.svm import SVC
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# 可以用cmd转变文件格式: dot -Tpdf car.dot -o car.pdf
# 这里有个小问题就是无法直接生成pdf,demo的库好像跟我版本不一样,明天再想办法
def plotTree(clf_tree, carSet):
    with open("car.dot", 'w') as f:
        tree.export_graphviz(clf_tree, out_file="car.dot",
                             feature_names=carSet.feature_names,
                             class_names=carSet.target_names,
                             filled=True, rounded=True,
                             special_characters=True)

def plotDecisionSurface(carSet, test_data, test_target):
    n_classes = 4
    plot_colors = "bryg"
    plot_step = 0.01

    # 随机添加一个随机浮点数,调节var可以控制变化幅度
    var = 0.3
    for i in range(len(carSet.target)):
        for j in range(6):
            carSet.data[i][j] = carSet.data[i][j] + random.uniform(-1*var, var)
    print (carSet.data)

    clf_tree = DecisionTreeClassifier( \
        criterion='entropy', splitter='best', \
        max_depth=None, min_samples_split=1, \
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
        max_features=None, random_state=42, \
        max_leaf_nodes=None, \
        class_weight=None, presort=False)
    
    #clf = DecisionTreeClassifier().fit(X, y)
    #SVM
    clf_svm=SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', \
                coef0=0.0, shrinking=True, probability=False,\
                 tol=0.001, cache_size=200, class_weight=None,\
                  verbose=False, max_iter=-1, \
                  decision_function_shape=None, random_state=None)
    #clf_tree = clf_tree.fit(X, y)
    #clf_svm=clf_svm.fit(X, y)
    classify=[clf_tree,clf_svm]

    #分别画出决策树和SVM的决策边界:
    tag=0
    for clf in classify:
        for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                                        [1, 2], [1, 3], [1, 4], [1, 5], [2, 3],
                                        [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]):
            # We only take the two corresponding features
            X = carSet.data[:, pair]
            y = carSet.target
            clf=clf.fit(X,y)

            # Plot the decision boundary
            plt.subplot(3, 5, pairidx + 1)
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            #print(x_min,x_max,y_min,y_max)
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
            plt.xlabel(carSet.feature_names[pair[0]])
            plt.ylabel(carSet.feature_names[pair[1]])
            plt.axis("tight")
            # Plot the training points
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(y == i)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, label=carSet.target_names[i],
                            cmap=plt.cm.Paired)
            plt.axis("tight")
            score = clf.score(test_data[:, pair], test_target)
            print("{0:2}:{1:<8} & {2:<10}".format(pairidx, carSet.feature_names[pair[0]], carSet.feature_names[pair[1]])+"准确率".ljust(7)+"{0}".format(score))
        print('*'*50)
        if tag==0:
            plt.suptitle("Decision surface of a decision tree using paired features")
        else:
            plt.suptitle("Decision surface of SVM using paired features")
        plt.legend()
        plt.show()
        tag+=1