import paddlenlp
import numpy as np


__all__ = ['RCInitVector', 'GloVeRC']


class RCInitVector:
    """
    Parent class of relation classification vector initializers.
    The input of relation classification problems include not only tokens of a sentence/text, but also the
    position of the entity in the sentence/text.
    For example, in the sentence ['Robin', 'works', 'for', 'Baidu'], the relation is 'work for' while the
    head entity is 'Robin' and the tail entity is 'Baidu'. As a result, the head_position is 0, the index of
    'Robin', and tail_position is 3, the index of 'Baidu'.

    """

    def __init__(self):
        super(RCInitVector, self).__init__()

    def __call__(self, tokens, head_position, tail_position, max_len=None):
        raise NotImplementedError("'{}' not implement in class {}".
                                  format('forward', self.__class__.__name__))


class GloVeRC(RCInitVector):
    """
    Relation classification vector initializer using pretrained GloVe embedding.
    This code will initialize relation classification inputs into vectors.

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
        'wiki': {
            50: 'glove.wiki2014-gigaword.target.word-word.dim50.en',
            100: 'glove.wiki2014-gigaword.target.word-word.dim100.en',
            200: 'glove.wiki2014-gigaword.target.word-word.dim200.en',
            300: 'glove.wiki2014-gigaword.target.word-word.dim300.en'
        },
        'twitter': {
            25: 'glove.twitter.target.word-word.dim25.en',
            50: 'glove.twitter.target.word-word.dim50.en',
            100: 'glove.twitter.target.word-word.dim100.en',
            200: 'glove.twitter.target.word-word.dim200.en'
        }
    }

    def __init__(self,
                 corpus='wiki',
                 embedding_dim=50):
        super(GloVeRC, self).__init__()
        if corpus not in self.embedding_file_name.keys():
            raise ValueError(
                'Corpus ' + corpus + ' not supported.\n' +
                'Paddle GloVe embedding support corpus: ' + str(list(self.embedding_file_name.keys()))
            )
        if embedding_dim not in self.embedding_file_name[corpus].keys():
            raise ValueError(
                'Embedding dimension ' + str(embedding_dim) + ' not supported.\n' +
                'Paddle GloVe ' + corpus + ' support dimension: ' + str(list(self.embedding_file_name[corpus].keys()))
            )
        file_name = self.embedding_file_name[corpus][embedding_dim]
        self.embedding = paddlenlp.embeddings.TokenEmbedding(file_name)

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
        embeddings = np.array(self.embedding.search(tokens))
        head_position, tail_position = min(max_len, head_position[0]), min(max_len, tail_position[0])
        head_position_idx = np.array([[i - head_position + max_len] for i in range(max_len)])
        tail_position_idx = np.array([[i - tail_position + max_len] for i in range(max_len)])
        rc_embedding = np.concatenate([embeddings, head_position_idx, tail_position_idx], axis=1)
        return rc_embedding
