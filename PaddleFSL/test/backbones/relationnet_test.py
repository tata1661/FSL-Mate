import paddle
from paddlefsl.backbones import ConvEmbedModel, ConvRelationModel


def conv_embed_model_test():
    train_input = paddle.ones(shape=(1, 1, 28, 28), dtype='float32')
    embed_model = ConvEmbedModel(input_size=(1, 28, 28), num_filters=64)
    print(embed_model.output_size)  # (64, 7, 7)
    print(embed_model(train_input))  # Tensor of shape [1, 64, 7, 7]


def conv_relation_model_test():
    prototypes = paddle.ones(shape=(5, 64, 21, 21), dtype='float32')  # 5: 5 ways
    query_embeddings = paddle.ones(shape=(10, 64, 21, 21), dtype='float32')  # 10: batch size 10
    embed_model = ConvRelationModel()
    print(embed_model(prototypes, query_embeddings))  # Tensor of shape [10, 5]


if __name__ == '__main__':
    conv_embed_model_test()
    conv_relation_model_test()
