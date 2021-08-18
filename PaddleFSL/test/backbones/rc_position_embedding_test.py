import paddle
from paddlefsl.backbones import RCPositionEmbedding


def rc_position_embedding_test():
    max_len = 100
    embedding_dim = 50
    position_embedding_dim = 5
    rc_vector = paddle.rand(shape=[5, max_len, embedding_dim + 2], dtype='float32')
    position_embedding = RCPositionEmbedding(
        max_len=max_len,
        embedding_dim=embedding_dim,
        position_embedding_dim=position_embedding_dim
    )
    print(position_embedding(rc_vector).shape)


if __name__ == '__main__':
    rc_position_embedding_test()
