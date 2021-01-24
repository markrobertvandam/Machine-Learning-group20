"""
Classifying the Digits dataset using a Random Forest Classifier
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


def main():
    x = []
    train_x = []
    test_x = []
    with open("mfeat-pix.txt", "r") as input_file:
        for line in input_file:
            x.append([int(pixel) for pixel in line.split()])
    indices = [i for i in range(0,2000,200)]
    for i in indices:
        train_x += x[i:i+100]
        test_x += x[i+100:i+200]
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    y = []
    for i in range(10):
        y += 100 * [i]
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=200, criterion = "gini", max_depth = None,
                                 min_samples_split = 4, min_samples_leaf = 1, min_weight_fraction_leaf = 0,
                                 max_features = "auto", max_leaf_nodes = None, min_impurity_decrease = 0.0,
                                 bootstrap = False, oob_score = False, n_jobs = -1, random_state = 1,
                                 verbose = 1, warm_start = False, class_weight = None, ccp_alpha = 0.0,
                                 max_samples = None)

    result = cross_val_score(clf, train_x, y, cv=10)
    print("crossval: ",np.average(result))

    #clf.fit(train_x, y)
    #print("Final score: ", clf.score(test_x, y))

if __name__ == "__main__":
    main()
