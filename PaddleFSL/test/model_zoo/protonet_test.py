import paddle
from paddlefsl.model_zoo import protonet


def get_prototype_test():
    support_embeddings = paddle.to_tensor([[1.1, 1.1, 1.1],
                                           [0.0, 0.0, 0.0],
                                           [0.9, 0.9, 0.9],
                                           [0.0, 0.0, 0.0]])
    support_labels = paddle.to_tensor([[1], [0], [1], [0]])
    prototypes = protonet.get_prototypes(support_embeddings, support_labels, ways=2, shots=2)
    print(prototypes)  # Tensor of [[0, 0, 0], [1, 1, 1]]


if __name__ == '__main__':
    get_prototype_test()
