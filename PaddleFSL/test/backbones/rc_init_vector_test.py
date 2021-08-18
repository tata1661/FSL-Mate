from paddlefsl.backbones import GloVeRC


def glove_rc_test():
    vector_initializer = GloVeRC()
    vector = vector_initializer(
        tokens=['yes', 'it', 'is', '*9*', '6$'],
        head_position=[0],
        tail_position=[2],
        max_len=6
    )
    print(vector)
    print(vector.shape)


if __name__ == '__main__':
    glove_rc_test()
