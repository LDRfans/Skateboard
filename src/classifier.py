import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle


def TestSVM(path="coco_pose.csv"):
    data = pd.read_csv(path, sep=",", header=0)
    data = shuffle(data)
    y = data.iloc[:240,4], data.iloc[240:,4]
    train_y, test_y = y
    x = data.iloc[:240,:4], data.iloc[240:,:4]
    train_x, test_x = x

    linear_svm = SVC(kernel='linear')
    linear_svm.fit(train_x, train_y)
    score_linear = linear_svm.score(train_x, train_y)
    print("Training Accuracy of linear svm is :", score_linear)
    score_linear_ = linear_svm.score(test_x, test_y)
    print("Testing Accuracy of linear svm is :", score_linear_)

    rbf_svm = SVC(kernel='rbf')
    rbf_svm.fit(train_x, train_y)
    score_rbf = rbf_svm.score(train_x, train_y)
    print("Trianing Accuracy of rbf svm is :", score_rbf)
    score_rbf_ = rbf_svm.score(test_x, test_y)
    print("Testing Accuracy of rbf svm is :", score_rbf_)

if __name__ == "__main__":
    TestSVM()