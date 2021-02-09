"""
Classifying the Digits dataset using a Random Forest Classifier
"""
import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def reduce_val_range(digits: np.ndarray) -> np.ndarray:
    """ Preprocess the digits dataset by reducing the value range of the pixels

    This method reduces the number of possible values that a pixel can have.
    Instead of a range from 0 through 6, a pixel will have a value of either
    0, 1 or 2. The mapping of [0 - 6] -> [0 - 2] is as follows:

    [0, 1] -> 0
    [2 - 4] -> 1
    [5, 6] -> 2

    :param digits: original digits dataset
    :return: digits dataset with a reduced pixel value range
    """
    return np.array([[round(pixel / 3) for pixel in digit] for digit in digits])


def downscale_by_factor(digits: np.ndarray, f: int) -> np.ndarray:
    """ Preprocess the digits dataset by downscaling it by a factor f

    We know the original size of the pictures is 15 x 16, so we should divide
    that number by the factor to get the number of rows and columns

    :param digits: original digits dataset
    :param f: integer factor to scale the the picture down by
    :return: digits dataset with a reduced pixel value range
    """
    new_digits = []
    for digit in digits:
        cols = int(np.ceil(15 / f))
        rows = int(np.ceil(16 / f))

        digit = np.array(np.array_split(digit, 16))

        new_digit = np.empty(rows * cols)

        # for each new pixel, take the average pixel value of a f x f block around it
        for y in range(rows):
            for x in range(cols):
                val = 0
                # sum each pixel's value
                for j in range(f):
                    for i in range(f):
                        kernel_y = y * f + j
                        kernel_x = x * f + i
                        val += digit[kernel_y][kernel_x] if kernel_y < 16 and kernel_x < 15 else 0
                # divide by the number of pixels summed
                new_digit[y * cols + x] = val / f ** 2

        new_digits.append(new_digit)

    return np.array(new_digits)


def read_digits_dataset(filename: str = "mfeat-pix.txt") -> np.ndarray:
    """ Open and read the digits dataset from the disk

    :return: digits dataset as a numpy array
    """
    x = []
    with open(filename, "r") as input_file:
        for line in input_file:
            x.append(np.array([int(pixel) for pixel in line.split()]))
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


def run_tuning(clf, digits) -> None:
    """ Run the random forest classifier to obtain cross-validation scores

    :param clf: random forest classifier instance
    :param digits: digits dataset, possibly preprocessed
    """
    score_raw = run_cross_val(clf, digits)
    print("No preprocessing score: {}".format(score_raw))
    score_reduced_range = run_cross_val(clf, reduce_val_range(digits))
    print("Reduced value range: {}".format(score_reduced_range))

    for i in range(2, 6):
        score_downscale = run_cross_val(clf, downscale_by_factor(digits, i))
        print("Downscale factor {}: {}".format(i, score_downscale))


def run_test(clf: RandomForestClassifier, x: np.ndarray) -> None:
    """ Run the random forest classifier to obtain the final test score

    :param clf: random forest classifier instance
    :param x: digits dataset, possibly preprocessed
    :return: test score
    """
    x_train, x_test = split_train_test(x)
    y = make_y()
    clf.fit(x_train, y)
    print("Final score: ", round(clf.score(x_test, y), 3))


def main() -> None:
    """ Run random forest machine learning"""
    digits = read_digits_dataset()
    classifier = RandomForestClassifier(
        n_estimators=200, criterion="gini", max_depth=None, min_samples_split=4, min_samples_leaf=1,
        min_weight_fraction_leaf=0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,
        bootstrap=False, oob_score=False, n_jobs=-1, random_state=1, verbose=0, warm_start=False,
        class_weight=None, ccp_alpha=0.0, max_samples=None
    )

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_test(classifier, reduce_val_range(digits))
        return

    run_tuning(classifier, digits)


if __name__ == "__main__":
    main()
