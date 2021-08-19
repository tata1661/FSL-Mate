import paddle
import paddlefsl
from paddlefsl.model_zoo import maml

# Set computing device
paddle.set_device('gpu:0')


# Config: MAML, Mini-ImageNet, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.MiniImageNet(mode='train')
VALID_DATASET = paddlefsl.datasets.MiniImageNet(mode='valid')
TEST_DATASET = paddlefsl.datasets.MiniImageNet(mode='test')
WAYS = 5
SHOTS = 1
MODEL = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=WAYS, conv_channels=[32, 32, 32, 32])
META_LR = 0.002
INNER_LR = 0.03
ITERATIONS = 60000
TEST_EPOCH = 10
META_BATCH_SIZE = 32
TRAIN_INNER_ADAPT_STEPS = 5
TEST_INNER_ADAPT_STEPS = 10
APPROXIMATE = True
REPORT_ITER = 10
SAVE_MODEL_ITER = 5000
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'iteration60000.params'


def main():
    train_dir = maml.meta_training(train_dataset=TRAIN_DATASET,
                                   valid_dataset=VALID_DATASET,
                                   ways=WAYS,
                                   shots=SHOTS,
                                   model=MODEL,
                                   meta_lr=META_LR,
                                   inner_lr=INNER_LR,
                                   iterations=ITERATIONS,
                                   meta_batch_size=META_BATCH_SIZE,
                                   inner_adapt_steps=TRAIN_INNER_ADAPT_STEPS,
                                   approximate=APPROXIMATE,
                                   report_iter=REPORT_ITER,
                                   save_model_iter=SAVE_MODEL_ITER,
                                   save_model_root=SAVE_MODEL_ROOT)
    print(train_dir)
    state_dict = paddle.load(train_dir + '/' + TEST_PARAM_FILE)
    MODEL.load_dict(state_dict)
    maml.meta_testing(model=MODEL,
                      test_dataset=TEST_DATASET,
                      test_epoch=TEST_EPOCH,
                      test_batch_size=META_BATCH_SIZE,
                      ways=WAYS,
                      shots=SHOTS,
                      inner_lr=INNER_LR,
                      inner_adapt_steps=TEST_INNER_ADAPT_STEPS,
                      approximate=APPROXIMATE)


if __name__ == '__main__':
    main()
