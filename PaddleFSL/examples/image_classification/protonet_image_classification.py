import paddle
import paddlefsl
from paddlefsl.model_zoo import protonet

# Set computing device
paddle.set_device('gpu:0')


""" ---------------------------------------------------------------------------------
# Config: ProtoNet, Omniglot, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.Omniglot(mode='train', image_size=(28, 28))
VALID_DATASET = paddlefsl.datasets.Omniglot(mode='valid', image_size=(28, 28))
TEST_DATASET = paddlefsl.datasets.Omniglot(mode='test', image_size=(28, 28))
WAYS = 5
SHOTS = 1
QUERY_NUM = 5
MODEL = paddlefsl.backbones.Conv(input_size=(1, 28, 28), output_size=WAYS)
MODEL.output = paddle.nn.Flatten()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR, parameters=MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100.params'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: ProtoNet, Omniglot, Conv, 5 Ways, 5 Shots
TRAIN_DATASET = paddlefsl.datasets.Omniglot(mode='train', image_size=(28, 28))
VALID_DATASET = paddlefsl.datasets.Omniglot(mode='valid', image_size=(28, 28))
TEST_DATASET = paddlefsl.datasets.Omniglot(mode='test', image_size=(28, 28))
WAYS = 5
SHOTS = 5
QUERY_NUM = 5
MODEL = paddlefsl.backbones.Conv(input_size=(1, 28, 28), output_size=WAYS)
MODEL.output = paddle.nn.Flatten()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR, parameters=MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100.params'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: ProtoNet, Mini-ImageNet, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.MiniImageNet(mode='train')
VALID_DATASET = paddlefsl.datasets.MiniImageNet(mode='valid')
TEST_DATASET = paddlefsl.datasets.MiniImageNet(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
MODEL = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=WAYS)
MODEL.output = paddle.nn.Flatten()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR, parameters=MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 20
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100.params'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: ProtoNet, CifarFS, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.CifarFS(mode='train')
VALID_DATASET = paddlefsl.datasets.CifarFS(mode='valid')
TEST_DATASET = paddlefsl.datasets.CifarFS(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
MODEL = paddlefsl.backbones.Conv(input_size=(3, 32, 32), output_size=WAYS)
MODEL.output = paddle.nn.Flatten()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR, parameters=MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100.params'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: ProtoNet, CifarFS, Conv, 5 Ways, 5 Shots
TRAIN_DATASET = paddlefsl.datasets.CifarFS(mode='train')
VALID_DATASET = paddlefsl.datasets.CifarFS(mode='valid')
TEST_DATASET = paddlefsl.datasets.CifarFS(mode='test')
WAYS = 5
SHOTS = 5
QUERY_NUM = 15
MODEL = paddlefsl.vision.backbones.Conv(input_size=(3, 32, 32), output_size=WAYS)
MODEL.output = paddle.nn.Flatten()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR, parameters=MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100.params'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: ProtoNet, FC100, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.FC100(mode='train')
VALID_DATASET = paddlefsl.datasets.FC100(mode='valid')
TEST_DATASET = paddlefsl.datasets.FC100(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
MODEL = paddlefsl.backbones.Conv(input_size=(3, 32, 32), output_size=WAYS)
MODEL.output = paddle.nn.Flatten()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.3)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR, parameters=MODEL.parameters())
EPOCHS = 50
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 5
SAVE_MODEL_EPOCH = 10
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch50.params'
----------------------------------------------------------------------------------"""


""" ---------------------------------------------------------------------------------
# Config: ProtoNet, CubFS, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.CubFS(mode='train')
VALID_DATASET = paddlefsl.datasets.CubFS(mode='valid')
TEST_DATASET = paddlefsl.datasets.CubFS(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
MODEL = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=WAYS)
MODEL.output = paddle.nn.Flatten()
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR, parameters=MODEL.parameters())
EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100.params'
----------------------------------------------------------------------------------"""


# """ ---------------------------------------------------------------------------------
# Config: ProtoNet, CubFS, Conv, 5 Ways, 5 Shots
root = '/home/wangyaqing/.cache/paddle/dataset'
TRAIN_DATASET = paddlefsl.datasets.CubFS(mode='train', root=root)
VALID_DATASET = paddlefsl.datasets.CubFS(mode='valid', root=root)
TEST_DATASET = paddlefsl.datasets.CubFS(mode='test', root=root)
WAYS = 5
SHOTS = 5
QUERY_NUM = 15
MODEL = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=WAYS)
MODEL.output = paddle.nn.Flatten()
LR = 0.012
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR, parameters=MODEL.parameters())
EPOCHS = 10
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = None
SAVE_MODEL_EPOCH = 10
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch10.params'
# ----------------------------------------------------------------------------------"""


def main():
    train_dir = protonet.meta_training(train_dataset=TRAIN_DATASET,
                                       valid_dataset=VALID_DATASET,
                                       model=MODEL,
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
    state_dict = paddle.load(train_dir + '/' + TEST_PARAM_FILE)
    MODEL.load_dict(state_dict)
    protonet.meta_testing(model=MODEL,
                          test_dataset=TEST_DATASET,
                          epochs=TEST_EPOCHS,
                          episodes=EPISODES,
                          ways=WAYS,
                          shots=SHOTS,
                          query_num=QUERY_NUM)


if __name__ == '__main__':
    main()
