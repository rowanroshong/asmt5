import argparse
import os
from collections import Counter, defaultdict

import numpy
from bs4 import BeautifulSoup as bs


class LexElt:
    def __init__(
        self,
        lexelt,
        skip_stopwords=False,
        do_casefolding=False,
        do_lemmas=False,
    ):
        self.key = lexelt["item"]
        self._instances = dict(
            [
                (i["id"], LexEltInstance(i))
                for i in lexelt.find_all("instance")
            ]
        )
        self.features = defaultdict(Counter)
        self.skip_stopwords = skip_stopwords
        self.do_casefolding = do_casefolding
        self.do_lemmas = do_lemmas

    def add_answer(self, instance_id, answers):
        self._instances[instance_id].add_answer(answers)

    def instances(self):
        return self._instances.values()

    def get(self, instance_id):
        return self._instances[instance_id]

    def keys(self):
        return self._instances.keys()

    def get_instance(self, instance_id):
        return self._instances[instance_id]

    # Start functions that students should write.
    def pos(self):
        ...

    def num_headwords(self):
        ...

    def num_answers(self):
        ...

    def get_all_senses(self):
        ...

    def count_unique_senses(self):
        ...

    def most_frequent_sense(self):
        ...

    def get_features(self, feature_names=None):
        ...

    def get_targets(self, labels=None):
        ...


class LexEltInstance:
    def __init__(self, instance):
        self.id = instance["id"]
        self.words = []
        self.heads = []
        self.doc = None
        self.answers = None
        self.features = Counter()

        for c in instance.context.contents:
            self.add_context(c)

    def add_context(self, c):
        if hasattr(c, "contents"):  # Head word
            text = c.contents[0]
            self.heads.append(len(self.words))
        else:  # Not a head word
            text = c.string.strip()
        self.words.extend(text.split())

    def add_answer(self, answers):
        self.answers = answers

    def has_answer(self, a):
        return a in self.answers

    # Start functions that students should write.
    def to_vector(self, feature_list):
        ...

    def bow_features(self):
        ...

    def colloc_features(self):
        ...

    def make_features(self):
        ...

    def get_feature_names(self):
        ...

    def bigrams(self):
        ...

    def trigrams(self):
        ...


def get_data(fp):
    """
    Input: input file pointer (training or test)

    Return: a dictionary mapping "lexelts" to LexElt objects.
    Each LexElt object stores its own instances.
    """

    soup = bs("<doc>{}</doc>".format(fp.read()), "xml")

    return dict(
        [
            (lexelt["item"], LexElt(lexelt))
            for lexelt in soup.findAll("lexelt")
        ]
    )


def get_key(fp, data):
    """Read the answer key"""
    for line in fp:
        fields = line.split()
        target = fields[0]
        instance_ID = fields[1]
        answers = fields[2:]
        data[target].add_answer(instance_ID, answers)


def main(args):
    trainexamplesfile = args.traindata
    testexamplesfile = args.testdata
    trainlabelfile = args.trainkey
    testlabelfile = args.testkey

    # train_fp = open(trainexamplesfile, "r")
    train_fp = open("C:/Users/rrros/OneDrive/Documents/COMPSCI/CMPU366/senseval3/train/EnglishLS.train", "r")
    trainkey_fp = open("C:/Users/rrros/OneDrive/Documents/COMPSCI/CMPU366/senseval3/train/EnglishLS.train.key", "r")
    train_data = get_data(train_fp)
    print(train_data.keys())
    print(train_data["smell.v"].keys())

    this_instance = train_data['smell.v'].get('smell.v.bnc.00018122')
    print(this_instance)
    print(" ".join(this_instance.words))
    heads = this_instance.heads
    print(this_instance.words[heads[0]])
    this_instance = train_data["smell.v"].get("smell.v.bnc.00006855")
    print(this_instance.heads)

    get_key(trainkey_fp, train_data)
    print(train_data["smell.v"].get('smell.v.bnc.00018122').answers)
    print(train_data["smell.v"].get("smell.v.bnc.00006855").answers)


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

    # Helping you out -- launch the debugger if something goes wrong!
    # import ipdb
    #
    # with ipdb.launch_ipdb_on_exception():
    #     main(args)
