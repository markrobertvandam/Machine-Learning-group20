"""
Classifying the Digits dataset using a Random Forest Classifier
"""
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def reduce_val_range(x: np.ndarray) -> np.ndarray:
    """ Preprocess the digits dataset by reducing the value range of the pixels

    This method reduces the number of possible values that a pixel can have.
    Instead of a range from 0 through 6, a pixel will have a value of either
    0, 1 or 2. The mapping of [0 - 6] -> [0 - 2] is as follows:

    [0, 1] -> 0
    [2 - 4] -> 1
    [5, 6] -> 2

    :param x: original digits dataset
    :return: digits dataset with a reduced pixel value range
    """
    return np.array([[round(pixel / 3) for pixel in digit] for digit in x])


def read_digits_dataset(filename: str = "mfeat-pix.txt") -> np.ndarray:
    """ Open and read the digits dataset from the disk

    :return: digits dataset as a numpy array
    """
    x = []
    with open(filename, "r") as input_file:
        for line in input_file:
            x.append([int(pixel) for pixel in line.split()])
    return np.array(x)


def split_train_test(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Split the digit dataset into two equal training and testing parts

    :param x: digits dataset
    :return: a tuple containing a training and testing datasets
    """
    x_train = []
    x_test = []
    x = x.tolist()
    for i in range(0, 2000, 200):
        x_train += x[i:i + 100]
        x_test += x[i + 100:i + 200]
    return np.array(x_train), np.array(x_test)


def make_y() -> np.ndarray:
    """ Create digit dataset labels

    :return: digit dataset labels
    """
    y = []
    for i in range(10):
        y += 100 * [i]
    return np.array(y)


def run_cross_val(clf: RandomForestClassifier, x: np.ndarray, k: int = 10) -> float:
    """ Run the random forest classifier to obtain a cross-validation score

    :param clf: random forest classifier instance
    :param x: digits dataset, possibly preprocessed
    :param k: number of folds to cross-validate with
    :return: k-fold x-val avg score
    """
    x_train, _ = split_train_test(x)
    return round(np.average(cross_val_score(clf, x_train, make_y(), cv=k)), 3)


def run_test(clf: RandomForestClassifier, x: np.ndarray) -> float:
    """ Run the random forest classifier to obtain the final test score

    :param clf: random forest classifier instance
    :param x: digits dataset, possibly preprocessed
    :return: test score
    """
    x_train, x_test = split_train_test(x)
    y = make_y()
    clf.fit(x_train, y)
    return round(clf.score(x_test, y), 3)


def main() -> None:
    """ Run random forest machine learning"""
    clf = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=None,
                                 min_samples_split=4, min_samples_leaf=1, min_weight_fraction_leaf=0,
                                 max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 bootstrap=False, oob_score=False, n_jobs=-1, random_state=1,
                                 verbose=1, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                 max_samples=None)

    x = read_digits_dataset()
    score_raw = run_cross_val(clf, x)
    score_reduced_range = run_cross_val(clf, reduce_val_range(x))
    print("No preprocessing avg x-val score: {}".format(score_raw))
    print("Reduced value range avg x-val score: {}".format(score_reduced_range))
    # print("test score: ", run_test(clf, x))


if __name__ == "__main__":
    main()
