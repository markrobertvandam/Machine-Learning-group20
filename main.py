"""
Classifying the Digits dataset using a Random Forest Classifier
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


def main():
    x = []
    with open("mfeat-pix.txt", "r") as input_file:
        for line in input_file:
            x.append([int(pixel) for pixel in line.split()])

    x = np.array(x)
    y = []
    for i in range(10):
        y += 200 * [i]
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=200)
    result = cross_val_score(clf, x, y, cv=10)
    print(np.average(result))


if __name__ == "__main__":
    main()
