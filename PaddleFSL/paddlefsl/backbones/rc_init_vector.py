# Copyright 2021 PaddleFSL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddlefsl.utils as utils
import numpy as np


__all__ = ['RCInitVector']


class RCInitVector:
    """
    Relation classification vector initializer using pretrained word embeddings.
    This code will initialize relation classification inputs into vectors. The input of relation classification
    problems include not only tokens of a sentence/text, but also the position of the entity in the sentence/text.
    For example, in the sentence ['Robin', 'works', 'for', 'Baidu'], the relation is 'work for' while the
    head entity is 'Robin' and the tail entity is 'Baidu'. As a result, the head_position is 0, the index of
    'Robin', and tail_position is 3, the index of 'Baidu'.
    We now only support embeddings of English tokens.

    Args:
        corpus(str, optional): corpus of pretrained GloVe. should be one of ['wiki', 'twitter'], default 'wiki'.
        embedding_dim(int, optional): embedding dimension of tokens, default 50.

    Examples:
        ..code-block:: python

            max_len = 128
            embedding_dim = 50
            init_vector = backbones.GloVeRC(embedding_dim=embedding_dim)
            TRAIN_DATASET = datasets.FewRel(mode='train', max_len=max_len, vector_initializer=init_vector)

    """

    embedding_file_name = {
        'glove-wiki': {
            50: 'glove.wiki2014-gigaword.target.word-word.dim50.en',
            100: 'glove.wiki2014-gigaword.target.word-word.dim100.en',
            200: 'glove.wiki2014-gigaword.target.word-word.dim200.en',
            300: 'glove.wiki2014-gigaword.target.word-word.dim300.en'
        },
        'glove-twitter': {
            25: 'glove.twitter.target.word-word.dim25.en',
            50: 'glove.twitter.target.word-word.dim50.en',
            100: 'glove.twitter.target.word-word.dim100.en',
            200: 'glove.twitter.target.word-word.dim200.en'
        },
        'fasttext-wiki': {
            300: 'fasttext.wiki-news.target.word-word.dim300.en'
        },
        'fasttext-crawl': {
            300: 'fasttext.crawl.target.word-word.dim300.en'
        },
        'word2vec': {
            300: 'w2v.google_news.target.word-word.dim300.en'
        }
    }
    url_head = 'https://paddlenlp.bj.bcebos.com/models/embeddings/'

    def __init__(self,
                 corpus='glove-wiki',
                 embedding_dim=50,
                 file_root=None):
        super(RCInitVector, self).__init__()
        self.embedding_dim = embedding_dim
        file_path = self._download_and_decompress(corpus, embedding_dim, file_root)
        vector_np = np.load(file_path)
        self.word2idx, self.embedding_table = self._init_word_list(vector_np)

    def _download_and_decompress(self, corpus, embedding_dim, file_root):
        if corpus not in self.embedding_file_name.keys():
            raise ValueError(
                'Corpus ' + corpus + ' not supported.\n' +
                'Support corpus: ' + str(list(self.embedding_file_name.keys()))
            )
        if embedding_dim not in self.embedding_file_name[corpus].keys():
            raise ValueError(
                'Embedding dimension ' + str(embedding_dim) + ' not supported.\n' +
                corpus + ' support dimension: ' + str(list(self.embedding_file_name[corpus].keys()))
            )
        file_name = self.embedding_file_name[corpus][embedding_dim]
        file_root = utils.process_root(file_root, 'word_vectors')
        if not utils.check_file_exist(file_root, file_name + '.npz'):
            file_path = utils.download_url(file_root, self.url_head + file_name + '.tar.gz')
            utils.decompress(file_path)
        return file_root + '/' + file_name + '.npz'

    def _init_word_list(self, vector_np):
        idx_to_word = list(vector_np['vocab']) + ['[UNK]', '[PAD]']
        word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
        # insert unk, pad embedding
        pad_vector = np.array([0] * self.embedding_dim)
        unk_vector = np.random.normal(scale=0.02, size=self.embedding_dim)
        embedding_table = np.append(vector_np['embedding'], [unk_vector, pad_vector], axis=0)
        return word_to_idx, embedding_table

    def get_idx_list_from_words(self, words):
        """
        Gets the index list of specifying words by searching word_to_idx dict.

        Args:
            words(str or List[str]): The input token words which we want to get the token indices converted from.

        Returns:
            List[int]: The indexes list of specifying words.

        Examples:
            ..code-block:: python

                vector_initializer = RCInitVector(corpus='glove-wiki', embedding_dim=50)
                idx_list = vector_initializer.get_idx_list_from_words('[PAD]')
                print(idx_list)  # [400001]
                idx_list = vector_initializer.get_idx_list_from_words(['i', 'love', 'you'])
                print(idx_list)  # [41, 835, 81]

        """
        if isinstance(words, str):
            idx_list = [self.word2idx[words]]
        elif isinstance(words, list):
            idx_list = [
                self.word2idx[word] if word in self.word2idx else self.word2idx['[UNK]']
                for word in words
            ]
        else:
            raise TypeError
        return idx_list

    def search_tokens(self, tokens):
        """
        Search for token embeddings of a list of tokens.

        Args:
            tokens(List[str]): a list of tokens to be embedded.

        Returns:
            numpy array of shape [len(tokens), embedding_dim]

        Examples:
            ..code-block:: python

                vector_initializer = RCInitVector(corpus='glove-wiki', embedding_dim=50)
                vector = vector_initializer.search_tokens(['i', 'love', 'robin', '[PAD]'])
                print(vector)
                print(vector.shape)  # (4, 50)

        """
        idx_list = self.get_idx_list_from_words(tokens)
        token_embeddings = self.embedding_table[idx_list]
        return token_embeddings

    def __len__(self):
        """
        Returns: capacity of the pretrained word embedding library, ['UNK'] and ['PAD'] excluded.

        """
        return self.embedding_table.shape[0] - 2

    def __call__(self, tokens, head_position, tail_position, max_len=None):
        """
        Call this function to initialize a single sentence/text.

        Args:
            tokens(List[str]): a list of tokens of the sentence/text.
            head_position(List[int]): position of the head entity of the relation.(An entity may have multiple tokens)
            tail_position(List[int]): position of the tail entity of the relation.
            max_len(int, optional): maximum length of the embedded sentence/text, default None. If not None, the
                sentence/text will be padded or truncated to fit max_len.

        Returns:
            numpy array of the embedded relation classification input sentence/text.
            shape: [max_len, embedding_dim + 2].
            Position information will be concatenated to the end of each token embedding.

        Examples:
            ..code-block:: python

                vector_initializer = GloVeRC()
                vector = vector_initializer(
                    tokens=['yes', 'it', 'is', '*9*', '6$'],
                    head_position=[0],
                    tail_position=[2],
                    max_len=6
                )
                print(vector)
                print(vector.shape)  # (6, 52)

        """
        max_len = len(tokens) if max_len is None else max_len
        if len(tokens) >= max_len:
            tokens = tokens[:max_len]
        else:
            tokens += ['[PAD]'] * (max_len - len(tokens))
        embeddings = np.array(self.search_tokens(tokens))
        head_position, tail_position = min(max_len, head_position[0]), min(max_len, tail_position[0])
        head_position_idx = np.array([[i - head_position + max_len] for i in range(max_len)])
        tail_position_idx = np.array([[i - tail_position + max_len] for i in range(max_len)])
        rc_embedding = np.concatenate([embeddings, head_position_idx, tail_position_idx], axis=1)
        return rc_embedding
