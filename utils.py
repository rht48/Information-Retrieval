import math


def tf_one_document(bag_of_words):
    tf = {}
    number_words = float(sum(bag_of_words.values()))
    for word in bag_of_words.keys():
        tf[word] = bag_of_words[word] / number_words
    return tf


def tf_documents(dict_documents):
    tf = {}
    for doc in dict_documents:
        tf[doc] = tf_one_document(dict_documents[doc])
    return tf


def idf_words(number_docs, inv_tokens):
    idf_dict = {}
    for word in inv_tokens.keys():
        idf_dict[word] = math.log(number_docs / float(len(inv_tokens[word].keys())))
        return idf_dict


def tf_idf_doc(tf_doc, idf_doc):
    tfidf = {}
    for word in tf_doc.keys():
        tfidf[word] = tf_doc[word] * idf_doc[word]
    return tfidf


def tf_idf(tokens, inv_tokens):
    tfidf = {}
    number_docs = len(tokens.keys())
    idf = idf_words(number_docs, inv_tokens)
    for doc in tokens.keys():
        tfidf[doc] = tf_idf_doc(tf_one_document(tokens[doc]), idf)
    return tfidf


