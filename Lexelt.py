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

    def pos(self):
        """
        Returns the part of speech of a LexElt object.
        type self: LexElt
        rtype: str
        """
        return self.key.split('.')[1]

    def num_headwords(self):
        """
        Returns a list with the number of headwords in each instance of a LexElt object.
        type self: LexElt
        rtype count_list: List[Int]
        """

        count_list = []
        for key in self.keys():
            count_list.append(len(self.get(key).heads))
        return count_list

    def num_answers(self):
        """
        Returns a list with the number of answers in each instance of a LexElt object.
        type self: LexElt
        rtype answer_list: List[Int]
        """
        count_list = []
        for key in self.keys():
            count_list.append(len(self.get(key).answers))
        return count_list


    def get_all_senses(self):
        """
        Returns a single list containing all of the senses of all of the instances of a LexElt object. Excludes "U" senses.
        Note that these can be listed either as numerical IDs or longer codes, depending on the data source.
        (Wordsmyth uses just an integer ID like "38201", while WordNet expresses metadata in the sense ID like
        "organization%1:04:02::".)
        type self: LexElt
        rtype senses_list: List[Int]
        """
        senses_list = []
        for key in self.keys():
            senses_list.append(self.get(key).answers)
        senses_list = [item for sublist in senses_list for item in sublist if item != 'U']
        return senses_list

    def count_unique_senses(self):
        """
        Returns the number of unique senses used by at least one of the instances of a LexElt object. Excludes "U".
        :return: The number of unique senses in the question.
        """
        return len(set(self.get_all_senses()))

    def most_frequent_sense(self):
        """
        Returns the most frequent sense of a word in a given context
        :return: The most frequent sense of the word.
        """
        maximum = max(self.get_all_senses(), key=self.get_all_senses().count)
        return maximum

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
    Given a file pointer to a training or test file, return a dictionary mapping lexelts to LexElt objects

    :param fp: the file pointer to the training or test data
    :return: A dictionary mapping lexelts to LexElt objects.
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
    # Part 1.1

    import Lexelt
    train_fp = open("/data/366/senseval3/train/EnglishLS.train", "r")
    train_data = get_data(train_fp)
    train_data.keys()
    # this_instance = train_data['smell.v'].get('smell.v.bnc.00018122')
    # heads = this_instance.heads
    # this_instance = train_data["smell.v"].get("smell.v.bnc.00006855")
    trainkey_fp = open("/data/366/senseval3/train/EnglishLS.train.key", "r")
    get_key(trainkey_fp, train_data)
    # print('hello')
    # print(train_data["smell.v"].get("smell.v.bnc.00018122").answers)
    # print(train_data["smell.v"].get("smell.v.bnc.00006855").answers)
    # lexelt = train_data["activate.n"]
    # print(lexelt.count_unique_senses())
    # print(train_data["activate.v"].answers)
    # print(lexelt.pos())
    # print(lexelt.num_headwords())
    # print(lexelt.num_answers())
    # print(lexelt.get_all_senses())
    # print(lexelt.count_unique_senses())
    # print(lexelt.most_frequent_sense())
    acc = 0.0
    total_headwords = 0
    for words in train_data.keys():
        lexelt = train_data[words]
        senses = lexelt.count_unique_senses()
        num_heads = sum(lexelt.num_headwords())
        total_headwords += num_heads
        acc +=  num_heads / senses
    print(acc/total_headwords)



if __name__ == "__main__":
    main('test')
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

    # Helping you out -- launch the debugger if something goes wrong!
    # import ipdb
    #
    # with ipdb.launch_ipdb_on_exception():
    #     main(args)
