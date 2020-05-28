# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import torch
import numpy as np
from ..common._registration import register_converter
from typing import List, Tensor


class CountVectorizer(torch.nn.Module):
    def __init__(self, vocabulary, max_features, stop_words, max_word_length, device):
        super(CountVectorizer, self).__init__()
        self.max_word_length = max_word_length
        self.original_vocab_size = max_features

        amp_factor = 1
        self.amp_factor = amp_factor

        bigrams_in_vocabluary = {key: vocabulary[key] for key in vocabulary if len(key.split(" ")) > 1}
        for bigram in bigrams_in_vocabluary:
            words = bigram.split(" ")
            for w in words:
                if w not in vocabulary:
                    vocabulary[w] = len(vocabulary)

        self.vocab_size_without_stop_words = len(vocabulary)

        for stop_word in stop_words:
            vocabulary[stop_word] = len(vocabulary)
        vocab_size = len(vocabulary)
        self.vocab_size = vocab_size

        words_in_vocabulary = {key: vocabulary[key] for key in vocabulary if len(key.split(" ")) == 1}

        # Word hashes
        collision = {}
        for w in words_in_vocabulary:
            val = self.__fast_hash_init(w, max_word_length, 0) % (vocab_size * amp_factor)
            if val in collision:
                collision[val].append(w)
            else:
                collision[val] = [w]

        max_collisions = 0
        for k in collision.keys():
            max_collisions = max(max_collisions, len(collision[k]))

        hash_table = np.zeros((vocab_size * amp_factor, max_collisions, max_word_length), dtype=np.uint8)
        feature_indices = np.zeros((vocab_size * amp_factor, max_collisions), dtype=np.int32)

        for k in collision.keys():
            for i, w in enumerate(collision[k]):
                hash_table[k][i][: min(self.max_word_length, len(w))] = np.array(
                    [ord(c) for c in w[: self.max_word_length]], dtype=np.uint8
                )
                feature_indices[k][i] = words_in_vocabulary[w] + 1  # zero is used for not in vocab

        hash_table = hash_table.view(np.int32).astype("int64")

        self.word_hash_table = torch.nn.Parameter(torch.from_numpy(hash_table), requires_grad=False).to(device)
        self.word_feature_indices = torch.nn.Parameter(torch.from_numpy(feature_indices), requires_grad=False).to(device)
        self.h = torch.nn.Parameter(torch.tensor([0], dtype=torch.int64), requires_grad=False)

        self.has_bigrams = False
        if len(bigrams_in_vocabluary) > 0:
            self.has_bigrams = True

            num_of_bigrams = len(bigrams_in_vocabluary)
            self.num_of_bigrams = num_of_bigrams
            collision = {}
            for w in bigrams_in_vocabluary:
                words = w.split(" ")
                k1, k2 = vocabulary[words[0]], vocabulary[words[1]]
                val = (k1 + k2 + 2) % (num_of_bigrams * self.amp_factor)

                if val in collision:
                    collision[val].append(w)
                else:
                    collision[val] = [w]

            max_collisions = 0
            for k in collision.keys():
                max_collisions = max(max_collisions, len(collision[k]))

            bigram_hash_table = -1 * np.ones((num_of_bigrams * self.amp_factor, max_collisions, 2), dtype=np.int64)
            bigram_feature_indices = np.zeros((num_of_bigrams * self.amp_factor, max_collisions), dtype=np.int32)

            for k in collision.keys():
                for i, w in enumerate(collision[k]):
                    words = w.split(" ")
                    k1, k2 = vocabulary[words[0]], vocabulary[words[1]]
                    bigram_hash_table[k][i][0] = k1 + 1
                    bigram_hash_table[k][i][1] = k2 + 1
                    bigram_feature_indices[k][i] = vocabulary[w] + 1  # zero is used for not in vocab

            self.bigram_hash_table = torch.nn.Parameter(torch.from_numpy(bigram_hash_table), requires_grad=False)
            self.bigram_feature_indices = torch.nn.Parameter(torch.from_numpy(bigram_feature_indices), requires_grad=False)

        self.powers_of_two = torch.nn.Parameter(torch.LongTensor([[[1, 2 ** 8, 2 ** 16, 2 ** 24]]]), requires_grad=False)

    @torch.jit.ignore
    def __fast_hash_init(self, s, max_word_length, seed=0):
        x = [ord(c) for c in list(s)]
        x = x[:max_word_length]

        while len(x) < max_word_length:
            x.append(0)

        length = len(x)
        c1 = 0xCC9E2D51
        c2 = 0x1B873593

        h = seed

        for pos in range(0, length, 4):
            b1 = 2 ** 24 * x[pos + 3] + 2 ** 16 * x[pos + 2] + 2 ** 8 * x[pos + 1] + x[pos + 0]
            b1 = (c1 * b1) & 0xFFFFFFFF
            b1 = (c2 * b1) & 0xFFFFFFFF

            h = h ^ b1

        return h

    def fast_hash_forward(self, x):
        max_size = self.max_word_length
        h = self.h

        c1 = 0xCC9E2D51
        c2 = 0x1B873593

        for pos in range(0, max_size // 4):
            b1 = x[:, pos]
            b1 = (c1 * b1) & 0xFFFFFFFF
            b1 = (c2 * b1) & 0xFFFFFFFF

            h = h ^ b1

        return h

    def forward(self, documents):
        # type: (List[Tensor]) -> Tensor

        doc_ids = torch.jit.annotate(List[Tensor], [])
        batch_size = len(documents)
        for i in range(len(documents)):
            doc_ids.append(i * torch.ones(documents[i].shape[0], dtype=torch.long))
        doc_ids = torch.cat(doc_ids)
        documents = torch.cat(documents)
        documents = documents.long()

        return self.forward_non_script(documents, doc_ids, torch.tensor(batch_size))

    @torch.jit.ignore
    def forward_non_script(self, documents, doc_ids, batch_size):
        documents = documents.view(-1, self.max_word_length // 4, 4)
        documents = (self.powers_of_two * documents).sum(dim=2, keepdim=False)

        hashes = self.fast_hash_forward(documents)
        indices = hashes % (self.vocab_size * self.amp_factor)
        lookup_values = torch.index_select(self.word_hash_table, 0, indices)
        lookup_indices = torch.index_select(self.word_feature_indices, 0, indices)

        indices = torch.eq(
            torch.eq(documents.view(-1, 1, self.max_word_length // 4), lookup_values).long().sum(2), self.max_word_length // 4
        )

        indices = (indices.int() * lookup_indices).sum(1)
        bin_count_indices = indices + (self.vocab_size + 1) * doc_ids
        feature_vector = torch.bincount(bin_count_indices, minlength=(self.vocab_size + 1) * batch_size)

        if self.has_bigrams:
            mask = indices < self.vocab_size_without_stop_words + 1
            indices = torch.masked_select(indices, mask)
            doc_ids = torch.masked_select(doc_ids, mask)

            bigram_candidates = torch.cat([indices[:-1].view(-1, 1), indices[1:].view(-1, 1)], dim=1)
            indices = torch.sum(bigram_candidates, dim=1) % (self.num_of_bigrams * self.amp_factor)

            lookup_values = torch.index_select(self.bigram_hash_table, 0, indices)
            lookup_indices = torch.index_select(self.bigram_feature_indices, 0, indices)

            indices = torch.eq(torch.eq(bigram_candidates.view(-1, 1, 2), lookup_values).long().sum(2), 2)
            indices = (indices.int() * lookup_indices).sum(1)

            doc_ids_mask = doc_ids[:-1] == doc_ids[1:]
            indices = torch.masked_select(indices, doc_ids_mask)
            doc_ids = torch.masked_select(doc_ids[:-1], doc_ids_mask)
            bin_count_indices = indices + (self.vocab_size + 1) * doc_ids

            feature_vector += torch.bincount(bin_count_indices, minlength=(self.vocab_size + 1) * batch_size)
            return feature_vector.view(-1, self.vocab_size + 1)[:, 1 : 1 + self.original_vocab_size]

        else:
            return feature_vector.view(-1, self.vocab_size + 1)[:, 1 : 1 + self.original_vocab_size]


def convert_sklearn_count_vectorizer(operator, device, extra_config):
    if max(operator.raw_operator.ngram_range) > 2 or operator.raw_operator.analyzer != "word":
        raise RuntimeError("skl2pytorch currently support only unigram and bigram word features")

    vocabulary = operator.raw_operator.vocabulary_

    max_features = operator.raw_operator.max_features
    if max_features is None:
        max_features = len(vocabulary)

    else:
        vocabulary = {word: vocabulary[word] for word in vocabulary if vocabulary[word] < max_features}

    # plus one to be safe with inference time words which are not seen before
    max_word_length = max([len(key) for key in vocabulary]) + 1
    while max_word_length % 4 != 0:
        max_word_length += 1

    if "skl2pytorch.CountVectorizer.max_word_length" in extra_config:
        if extra_config["skl2pytorch.CountVectorizer.max_word_length"] < max_word_length:
            raise RuntimeError("The provided skl2pytorch.CountVectorizer.max_word_length is inconsistent with the vocabulary")
        elif extra_config["skl2pytorch.CountVectorizer.max_word_length"] % 4 != 0:
            raise RuntimeError("skl2pytorch.CountVectorizer.max_word_length has to be a multiple of 4")
        else:
            max_word_length = extra_config["skl2pytorch.CountVectorizer.max_word_length"]

    if operator.raw_operator.stop_words is not None:
        stop_words = operator.raw_operator.stop_words
    else:
        stop_words = []

    return torch.jit.script(CountVectorizer(vocabulary, max_features, stop_words, max_word_length, device))


register_converter("SklearnCountVectorizer", convert_sklearn_count_vectorizer)
