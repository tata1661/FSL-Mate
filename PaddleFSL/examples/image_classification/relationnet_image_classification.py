import paddle
import paddlefsl
from paddlefsl.model_zoo import relationnet


# Set computing device
paddle.set_device('gpu:1')


""" ---------------------------------------------------------------------------------
# Config: RelationNet, Omniglot, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.Omniglot(mode='train', image_size=(28, 28))
VALID_DATASET = paddlefsl.datasets.Omniglot(mode='valid', image_size=(28, 28))
TEST_DATASET = paddlefsl.datasets.Omniglot(mode='test', image_size=(28, 28))
WAYS = 5
SHOTS = 1
QUERY_NUM = 19
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(1, 28, 28))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.0001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: RelationNet, Omniglot, Conv, 5 Ways, 5 Shots
TRAIN_DATASET = paddlefsl.datasets.Omniglot(mode='train', image_size=(28, 28))
VALID_DATASET = paddlefsl.datasets.Omniglot(mode='valid', image_size=(28, 28))
TEST_DATASET = paddlefsl.datasets.Omniglot(mode='test', image_size=(28, 28))
WAYS = 5
SHOTS = 5
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(1, 28, 28))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.0001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 20
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch20'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: RelationNet, Mini-ImageNet, Conv, 5 Ways, 1 Shot
root = '/home/wangyaqing/.cache/paddle/dataset'
TRAIN_DATASET = paddlefsl.datasets.MiniImageNet(mode='train', root=root)
VALID_DATASET = paddlefsl.datasets.MiniImageNet(mode='valid', root=root)
TEST_DATASET = paddlefsl.datasets.MiniImageNet(mode='test', root=root)
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 84, 84))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.002, gamma=0.7)
LR_STEP_EPOCH = 20
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 200
TEST_EPOCHS = 10
EPISODES = 600
REPORT_EPOCH = 1
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch200'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: RelationNet, Mini-ImageNet, Conv, 5 Ways, 5 Shots
root = '/home/wangyaqing/.cache/paddle/dataset'
TRAIN_DATASET = paddlefsl.datasets.MiniImageNet(mode='train', root=root)
VALID_DATASET = paddlefsl.datasets.MiniImageNet(mode='valid', root=root)
TEST_DATASET = paddlefsl.datasets.MiniImageNet(mode='test', root=root)
WAYS = 5
SHOTS = 5
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 84, 84))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.002, gamma=0.8)
LR_STEP_EPOCH = 10
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 600
REPORT_EPOCH = 1
SAVE_MODEL_EPOCH = 10
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch90'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: RelationNet, CifarFS, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.CifarFS(mode='train')
VALID_DATASET = paddlefsl.datasets.CifarFS(mode='valid')
TEST_DATASET = paddlefsl.datasets.CifarFS(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 32, 32))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: RelationNet, CifarFS, Conv, 5 Ways, 5 Shots
TRAIN_DATASET = paddlefsl.datasets.CifarFS(mode='train')
VALID_DATASET = paddlefsl.datasets.CifarFS(mode='valid')
TEST_DATASET = paddlefsl.datasets.CifarFS(mode='test')
WAYS = 5
SHOTS = 5
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 32, 32))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: RelationNet, FC100, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.FC100(mode='train')
VALID_DATASET = paddlefsl.datasets.FC100(mode='valid')
TEST_DATASET = paddlefsl.datasets.FC100(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 32, 32))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 30
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 10
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch30'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: RelationNet, FC100, Conv, 5 Ways, 5 Shots
TRAIN_DATASET = paddlefsl.datasets.FC100(mode='train')
VALID_DATASET = paddlefsl.datasets.FC100(mode='valid')
TEST_DATASET = paddlefsl.datasets.FC100(mode='test')
WAYS = 5
SHOTS = 5
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 32, 32))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.0002, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 10
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
Config: RelationNet, CubFS, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.CubFS(mode='train')
VALID_DATASET = paddlefsl.datasets.CubFS(mode='valid')
TEST_DATASET = paddlefsl.datasets.CubFS(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 84, 84))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=2e-4, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 50
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 10
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch50'
----------------------------------------------------------------------------------"""


# """ ---------------------------------------------------------------------------------
# Config: RelationNet, CubFS, Conv, 5 Ways, 5 Shots
root = '~/zhaozijing/dataset'
TRAIN_DATASET = paddlefsl.datasets.CubFS(mode='train', root=root)
VALID_DATASET = paddlefsl.datasets.CubFS(mode='valid', root=root)
TEST_DATASET = paddlefsl.datasets.CubFS(mode='test', root=root)
WAYS = 5
SHOTS = 5
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 84, 84))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=2e-4, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 50
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 10
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch50'
# ----------------------------------------------------------------------------------"""


def main():
    train_dir = relationnet.meta_training(train_dataset=TRAIN_DATASET,
                                          valid_dataset=VALID_DATASET,
                                          embedding_model=EMBEDDING_MODEL,
                                          relation_model=RELATION_MODEL,
                                          lr=LR,
                                          optimizer=OPTIMIZER,
                                          epochs=EPOCHS,
                                          episodes=EPISODES,
                                          ways=WAYS,
                                          shots=SHOTS,
                                          query_num=QUERY_NUM,
                                          report_epoch=REPORT_EPOCH,
                                          lr_step_epoch=LR_STEP_EPOCH,
                                          save_model_epoch=SAVE_MODEL_EPOCH,
                                          save_model_root=SAVE_MODEL_ROOT)
    print(train_dir)
    state_dict = paddle.load(train_dir + '/' + TEST_PARAM_FILE + '_embedding.params')
    EMBEDDING_MODEL.load_dict(state_dict)
    state_dict = paddle.load(train_dir + '/' + TEST_PARAM_FILE + '_relation.params')
    RELATION_MODEL.load_dict(state_dict)
    for i in range(5):
        relationnet.meta_testing(embedding_model=EMBEDDING_MODEL,
                                 relation_model=RELATION_MODEL,
                                 test_dataset=TEST_DATASET,
                                 epochs=TEST_EPOCHS,
                                 episodes=EPISODES,
                                 ways=WAYS,
                                 shots=SHOTS,
                                 query_num=QUERY_NUM)


if __name__ == '__main__':
    main()
