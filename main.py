import nltk
import re
from nltk.tokenize import RegexpTokenizer
from utils import *

nltk.download('punkt')


def keep(word):
    # return word not in stopwords
    return True


def is_new_document(line):
    m = re.search(r".I \d+", line)
    return m is not None


def tokenize_file(filename):
    file = open(filename, "r")
    document = {}
    count = 0
    tokenizer = RegexpTokenizer(r'\w+')
    for line in file:
        if not is_new_document(line):
            for word in tokenizer.tokenize(line.lower()):
                # word = lemmatize(word)
                # word = stem(word)
                if keep(word):
                    if word not in document[count]:
                        document[count].update({word: 0})
                    document[count][word] = document[count][word] + 1
        else:
            count = count + 1
            document.update({count: {}})
    return document


def inverse_tokens(toks):
    inv = {}
    for doc in toks:
        for word in toks[doc]:
            if word not in inv.keys():
                inv.update({word: {}})
            inv[word].update({doc: toks[doc][word]})
    return inv


if __name__ == "__main__":
    tokens = tokenize_file("CISI.ALLnettoye")
    inv_tokens = inverse_tokens(tokens)
    print(tokens)
    print()
    print(inv_tokens)
    tfidf = tf_idf(tokens, inv_tokens)
    print(tfidf)