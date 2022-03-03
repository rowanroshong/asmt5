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
        # Convert to numpy array so we can index from it
        features_indices_list = numpy.asarray(feature_indices)[0]

        sense_at_feature_indices = y[feature_indices]
        indices_sense_at = numpy.where(sense_at_feature_indices == label)
        indices_sense_not_at = numpy.where(sense_at_feature_indices != label)
        X_with_correct_sense = features_indices_list[indices_sense_at]
        X_without_correct_sense = features_indices_list[indices_sense_not_at]

        has_correct_sense = X[X_with_correct_sense, feature]
        not_correct_sense = X[X_without_correct_sense, feature]

        correct_sense = sum(has_correct_sense)
        incorrect_sense = sum(not_correct_sense)

        correct_sense += self.alpha
        incorrect_sense += self.alpha

        score = math.log((correct_sense/incorrect_sense), 2)

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
        indices = numpy.where(vector > 0)
        indices = numpy.asarray(indices)[0]
        for e in self.rules:
            if e[0] in indices:
                return e[1]
        return self.default_target



    def predict(self, X):
        length = len(X)
        array = numpy.zeros(length)
        for i, vector in enumerate(X):
            array[i] = self.predict_one(vector)
        return array

def main(args):
    # train_data = get_data(args.traindata)
    # get_key(args.trainkey, train_data)
    #
    # test_data = get_data(args.testdata)
    # get_key(args.testkey, test_data)

    # train_fp = open("C:/Users/rrros/OneDrive/Documents/COMPSCI/CMPU366/senseval3/train/EnglishLS.train", "r")
    train_fp = open("/data/366/senseval3/train/EnglishLS.train", "r")
    train_data = get_data(train_fp)
    train_data.keys()
    # trainkey_fp = open("C:/Users/rrros/OneDrive/Documents/COMPSCI/CMPU366/senseval3/train/EnglishLS.train.key", "r")
    trainkey_fp = open("/data/366/senseval3/train/EnglishLS.train.key", "r")
    get_key(trainkey_fp, train_data)

    d = DecisionList(alpha=0.1, min_score=5, default_target=5)

    lexelt = train_data['activate.v']
    this_instance = lexelt.get("activate.v.bnc.00044852")
    this_instance.make_features()
    feature_names, X_train = lexelt.get_features()
    answer_labels, Y_train = lexelt.get_targets()

    print(d.get_score(878, 1, X_train, Y_train))
    print(d.get_score(4482, 0, X_train, Y_train))
    d.fit(X_train, Y_train)
    print(d.rules[:10])

    d2 = DecisionList(alpha=0.1, min_score=5, default_target=5)
    test_fp = open("/data/366/senseval3/test/EnglishLS.test", "r")
    test_data = get_data(test_fp)
    lexelt2 = test_data['activate.v']
    testkey_fp = open("/data/366/senseval3/test/EnglishLS.test.key", "r")
    get_key(testkey_fp, test_data)
    test_feature_names, X_test = lexelt2.get_features()
    test_answer_labels, Y_test = lexelt2.get_targets()

    d2.fit(X_test, Y_test)
    print(d2.predict_one(X_test[0]))
    print(d2.predict_one(X_test[101]))
    print(d2.predict(X_test))

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
