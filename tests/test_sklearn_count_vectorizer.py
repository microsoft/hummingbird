"""
Tests sklearn Normalizer converter
"""
import unittest
import warnings

import nltk
import numpy as np
import re
import torch

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import hummingbird.ml

nltk.download("stopwords")


class TestSklearnCountVectorizerConverter(unittest.TestCase):
    def test_count_vectorizer_converter(self):

        text_documents = fetch_20newsgroups(subset="all", shuffle=True).data[:1000]

        cleaner = re.compile("[^a-zA-Z\s\.\@]")  # noqa: W605
        text_documents = [cleaner.sub("", doc.encode("ascii", "replace").decode("ascii")) for doc in text_documents]

        ## tokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b")

        model = CountVectorizer(max_features=10000, stop_words=stopwords.words("english"), ngram_range=(1, 2))
        model.fit(text_documents)

        max_word_length = (
            max(
                [
                    len(key)
                    for key in model.vocabulary_
                    if model.max_features is not None and model.vocabulary_[key] < model.max_features
                ]
            )
            + 1
        )
        while max_word_length % 4 != 0:
            max_word_length += 1

        torch_model = hummingbird.ml.convert(model, "torch")

        self.assertTrue(torch_model is not None)
        # TODO:fixup input data
        ### data_tensor = [
        ###     torch.from_numpy(np.array(tokenizer.tokenize(doc.lower()), dtype='|S'+str(max_word_length)).view(np.uint8)).view(-1, max_word_length) for doc
        ###     in text_documents]
        ### self.assertTrue(np.allclose(model.transform(text_documents).todense(), torch_model.transform(data_tensor)))


if __name__ == "__main__":
    unittest.main()
