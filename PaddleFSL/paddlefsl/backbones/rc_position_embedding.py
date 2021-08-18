import paddle
import paddle.nn as nn


class RCPositionEmbedding(nn.Layer):
    """
    Relation classification position embedding module.
    This module embeds the position information in relation classification.

    Args:
        max_len(int, optional): maximum length of a single sentence/text, default 100.
        embedding_dim(int, optional): token embedding dimension, default 50.
        position_embedding_dim(int, optional): position embedding dimension, default 5.

    """

    def __init__(self, max_len=100, embedding_dim=50, position_embedding_dim=5):
        super(RCPositionEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.position_embedding_dim = position_embedding_dim
        self.head_position_embedding = nn.Embedding(max_len * 2, position_embedding_dim, padding_idx=0)
        self.tail_position_embedding = nn.Embedding(max_len * 2, position_embedding_dim, padding_idx=0)
        self.embedding_size = embedding_dim + 2 * position_embedding_dim

    def forward(self, rc_vector):
        """
        For each vector, v[:token_embedding_dim] represents token and will not be changed in this
                function, while v[-2] amd v[-1] represents head entity position and tail entity position.
                Position information will be embedded and concatenated to the end of each token embedding.

        Args:
            rc_vector: relation classification batch vector. shape: [batch_size, token_embedding_dim + 2].

        Returns:
            shape [batch_size, token_embedding_dim + 2 * position_embedding_dim]

        Examples:
            ..code-block:: python

                max_len = 100
                embedding_dim = 50
                position_embedding_dim = 5
                rc_vector = paddle.rand(shape=[5, max_len, embedding_dim + 2], dtype='float32')
                position_embedding = RCPositionEmbedding(
                    max_len=max_len,
                    embedding_dim=embedding_dim,
                    position_embedding_dim=position_embedding_dim
                )
                print(position_embedding(rc_vector).shape)  # (5, 100, 60)

        """
        if rc_vector.shape[-1] != self.embedding_dim + 2:
            raise ValueError('Embedding dimension not match. Please check whether'
                             'embedding_dimension + 2 == rc_vector.shape[-1]')
        head_position_vector = paddle.to_tensor(rc_vector[:, :, -1], dtype='int64')
        head_embedding = self.head_position_embedding(head_position_vector)
        tail_position_vector = paddle.to_tensor(rc_vector[:, :, -1], dtype='int64')
        tail_embedding = self.head_position_embedding(tail_position_vector)
        output = paddle.concat([rc_vector[:, :, :self.embedding_dim], head_embedding, tail_embedding], axis=-1)
        return output
