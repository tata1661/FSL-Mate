# Copyright 2021 PaddleFSL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
