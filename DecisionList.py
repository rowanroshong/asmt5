#!/usr/bin/env python3

import argparse
import math
import numpy
from Lexelt import *


class DecisionList(object):
    def __init__(self, default_target=None, alpha=0.1, min_score=0):
        """Create a DecisionList classifier"""
        self.alpha = alpha
        self.default_target = default_target
        self.rules = []
        self.min_score = min_score

    def get_score(self, feature, label, X, y):
        feature_indices = numpy.where(X[:, feature])

        x = y[feature_indices]

        indices = numpy.where(x == label)

        values = x[indices]
        # print(values)

        summed_values = len(values)
        without_feature = len(x) - summed_values

        summed_values += self.alpha
        without_feature += self.alpha

        score = math.log((summed_values/without_feature), 2)

        # print(score)

        return score

    def fit(self, X, y):
        pairs_with_score = []

        # Get unique and convert to int
        y_set = set(y)
        y_set = [int(item) for item in y_set]
        num_features = len(X[0])

        for label in y_set:
            for k in range(0, num_features):
                score = self.get_score(k, label, X, y)
                if score >= self.min_score:
                    pairs_with_score.append(((k, label), score))

        # Sort by score
        pairs_with_score.sort(key=lambda x: x[1], reverse=True)
        # Get just the pairs
        pairs = [a_tuple[0] for a_tuple in pairs_with_score]
        self.rules = pairs

    def predict_one(self, vector):
        ...

    def predict(self, X):
        ...


def main(args):
    # train_data = get_data(args.traindata)
    # get_key(args.trainkey, train_data)
    #
    # test_data = get_data(args.testdata)
    # get_key(args.testkey, test_data)

    train_fp = open("C:/Users/rrros/OneDrive/Documents/COMPSCI/CMPU366/senseval3/train/EnglishLS.train", "r")
    # train_fp = open("/data/366/senseval3/train/EnglishLS.train", "r")
    train_data = get_data(train_fp)
    train_data.keys()
    trainkey_fp = open("C:/Users/rrros/OneDrive/Documents/COMPSCI/CMPU366/senseval3/train/EnglishLS.train.key", "r")
    # trainkey_fp = open("/data/366/senseval3/train/EnglishLS.train.key", "r")
    get_key(trainkey_fp, train_data)

    d = DecisionList(alpha=0.1, min_score=5, default_target=5)

    lexelt = train_data['activate.v']
    this_instance = lexelt.get("activate.v.bnc.00044852")
    this_instance.make_features()
    feature_names, X_train = lexelt.get_features()
    answer_labels, Y_train = lexelt.get_targets()

    print(d.get_score(0, 1, X_train, Y_train))
    d.fit(X_train, Y_train)
    print(d.rules[:10])

    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traindata",
        type=argparse.FileType("r"),
        help="xml file containing training examples",
    )
    parser.add_argument(
        "--testdata",
        type=argparse.FileType("r"),
        help="xml file containing test examples",
    )
    parser.add_argument(
        "--trainkey",
        type=argparse.FileType("r"),
        help="file containing training labels",
    )
    parser.add_argument(
        "--testkey",
        type=argparse.FileType("r"),
        help="file containing test labels",
    )

    args = parser.parse_args()
    main(args)
