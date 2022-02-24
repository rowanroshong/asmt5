#!/usr/bin/env python3

import argparse
import math

import numpy

from Lexelt import get_data, get_key


class DecisionList(object):
    def __init__(self, default_target=None, alpha=0.1, min_score=0):
        """Create a DecisionList classifier"""
        self.alpha = alpha
        self.default_target = default_target
        self.rules = []
        self.min_score = min_score

    def get_score(self, feature, label, X, y):
        ...

    def fit(self, X, y):
        ...

    def predict_one(self, vector):
        ...

    def predict(self, X):
        ...


def main(args):
    train_data = get_data(args.traindata)
    get_key(args.trainkey, train_data)

    test_data = get_data(args.testdata)
    get_key(args.testkey, test_data)

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
