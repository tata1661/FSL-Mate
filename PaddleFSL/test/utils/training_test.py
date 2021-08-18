import paddle
import paddlefsl
import paddlefsl.utils as utils


def get_info_str_test():
    max_len, embedding_dim = 100, 50
    position_embedding = paddlefsl.backbones.RCPositionEmbedding(max_len=max_len, embedding_dim=embedding_dim)
    conv_model = paddlefsl.backbones.RCConv1D(max_len=max_len, embedding_size=position_embedding.embedding_size)
    model = paddle.nn.Sequential(
        position_embedding,
        conv_model
    )
    model._full_name = 'glove50_cnn'
    ways = 5
    shots = 5
    info_str = utils.get_info_str(model, ways, 'ways', shots, 'shots')
    print(info_str)  # 'conv_5_ways_5_shots'


def print_training_info_test():
    train_loss, train_acc = 0.85, 0.76
    utils.print_training_info(0, train_loss, train_acc, info='just a test')
    # 'Iteration 0	just a test'
    # 'Training Loss 0.85	Training Accuracy 0.76'


if __name__ == '__main__':
    get_info_str_test()
    print_training_info_test()
