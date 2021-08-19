from paddlefsl.backbones import RCInitVector

vector_initializer = RCInitVector(corpus='glove-wiki', embedding_dim=50)


def get_idx_list_from_words_test():
    idx_list = vector_initializer.get_idx_list_from_words('[PAD]')
    print(idx_list)
    idx_list = vector_initializer.get_idx_list_from_words(['i', 'love', 'you'])
    print(idx_list)


def search_tokens_test():
    vector = vector_initializer.search_tokens(['i', 'love', 'robin', '[PAD]'])
    print(vector)
    print(vector.shape)


def rc_init_vector_test():
    vector = vector_initializer(
        tokens=['yes', 'it', 'is', '*9*', '6$'],
        head_position=[0],
        tail_position=[2],
        max_len=6
    )
    print(len(vector_initializer))
    print(vector)
    print(vector.shape)


if __name__ == '__main__':
    get_idx_list_from_words_test()
    search_tokens_test()
    rc_init_vector_test()
