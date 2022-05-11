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

import os
import json
import numpy as np

import paddle
from paddlenlp.utils.log import logger

#TODO
def clue_pet_preprocess(example,
                       args,  
                       task_name,
                       is_test = False):
    """
    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        p_embedding_num(obj:`int`) The number of p-embedding.
    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
        mask_positions(obj: `list[int]`): The list of mask_positions.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
    """

    example = transform_pet_fn_dict[task_name] (example, label_normalize_dict = args.label_dict, \
                                                 pattern_id=args.pattern_id)
    tokenizer = args.tokenizer
    max_seq_length = args.max_seq_length

    # Replace <unk> with '[MASK]'

    # Step1: gen mask ids
    if is_test:
        label_length = example["label_length"]
    else:
        text_label = example["text_label"]
        label_length = len(text_label)

    mask_tokens = ["[MASK]"] * label_length
    mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)

    sentence1 = example["sentence1"]
    if "<unk>" in sentence1:
        start_mask_position = sentence1.index("<unk>") + 1
        sentence1 = sentence1.replace("<unk>", "")
        encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
        src_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]

        # Step2: Insert "[MASK]" to src_ids based on start_mask_position
        src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
            start_mask_position:]
        token_type_ids = token_type_ids[0:start_mask_position] + [0] * len(
            mask_ids) + token_type_ids[start_mask_position:]

        # calculate mask_positions
        mask_positions = [
            index + start_mask_position for index in range(label_length)
        ]
    else:
        sentence2 = example['sentence2']
        start_mask_position = sentence2.index("<unk>") + 1
        sentence2 = sentence2.replace("<unk>", "")

        encoded_inputs = tokenizer(text=sentence2, max_seq_len=max_seq_length)
        src_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
            start_mask_position:]
        token_type_ids = token_type_ids[0:start_mask_position] + [0] * len(
            mask_ids) + token_type_ids[start_mask_position:]

        encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
        sentence1_src_ids = encoded_inputs["input_ids"][1:]
        src_ids = sentence1_src_ids + src_ids
        token_type_ids += [1] * len(src_ids)
        mask_positions = [
            index + start_mask_position + len(sentence1)
            for index in range(label_length)
        ]

    token_type_ids = [0] * len(src_ids)

    assert len(src_ids) == len(
        token_type_ids), "length src_ids, token_type_ids must be equal"

    length = len(src_ids)
    if length > 512:
        src_ids = src_ids[:512]
        token_type_ids = token_type_ids[:512]

    if is_test:
        return src_ids, token_type_ids, mask_positions
    else:
        mask_lm_labels = tokenizer(
            text=text_label, max_seq_len=max_seq_length)["input_ids"][1:-1]

        assert len(mask_lm_labels) == len(
            mask_positions
        ) == label_length, "length of mask_lm_labels:{} mask_positions:{} label_length:{} not equal".format(
            mask_lm_labels, mask_positions, text_label)
        return src_ids, token_type_ids, mask_positions, mask_lm_labels


def chid_pet_preprocess(example,
                       args,  
                       task_name,
                       is_test = False):
    """
    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        p_embedding_num(obj:`int`) The number of p-embedding.
    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
        mask_positions(obj: `list[int]`): The list of mask_positions.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
    """

    example = transform_pet_fn_dict[task_name] (example, label_normalize_dict = args.label_dict, \
                                                 pattern_id=args.pattern_id)
    tokenizer = args.tokenizer
    max_seq_length = args.max_seq_length

    # FewClue Task `Chid`' label's position must be calculated by special token: "淠"
    seg_tokens = tokenizer.tokenize(example["sentence1"])

    # find insert position of `[MASK]`
    start_mask_position = seg_tokens.index("淠") + 1
    seg_tokens.remove("淠")

    sentence1 = "".join(seg_tokens)
    candidates = example["candidates"]
    candidate_labels_ids = [
        tokenizer(text=idom)["input_ids"][1:-1] for idom in candidates
    ]

    sentence1 = example["sentence1"]

    encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
    src_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    # Step1: gen mask ids
    if is_test:
        label_length = example["label_length"]
    else:
        text_label = example["text_label"]
        label_length = len(text_label)

    mask_tokens = ["[MASK]"] * label_length
    mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)

    # Step2: Insert "[MASK]" to src_ids based on start_mask_position
    src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
        start_mask_position:]
    token_type_ids = token_type_ids[0:start_mask_position] + [0] * len(
        mask_ids) + token_type_ids[start_mask_position:]

    # calculate mask_positions
    mask_positions = [
        index + start_mask_position for index in range(label_length)
    ]

    # token_type_ids = [0] * len(src_ids)

    assert len(src_ids) == len(
        token_type_ids), "length src_ids, token_type_ids must be equal"

    length = len(src_ids)
    if length > 512:
        src_ids = src_ids[:512]
        token_type_ids = token_type_ids[:512]

    if is_test:
        return src_ids, token_type_ids, mask_positions, candidate_labels_ids
    else:
        mask_lm_labels = tokenizer(
            text=text_label, max_seq_len=max_seq_length)["input_ids"][1:-1]

        assert len(mask_lm_labels) == len(
            mask_positions
        ) == label_length, "length of mask_lm_labels:{} mask_positions:{} label_length:{} not equal".format(
            mask_lm_labels, mask_positions, text_label)

        return src_ids, token_type_ids, mask_positions, mask_lm_labels, candidate_labels_ids


def transform_pet_iflytek(example,
                      label_normalize_dict=None,
                      is_test=False,
                      pattern_id=0):

    if is_test:
        # When do_test, set label_length field to point
        # where to insert [MASK] id
        example["label_length"] = 2

        if pattern_id == 0:
            example["sentence1"] = u'作为一款<unk>应用，' + example["sentence"]
        elif pattern_id == 1:
            example["sentence1"] = u'这是一款<unk>应用！' + example["sentence"]
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'。 和<unk>有关'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>!'

        return example
    else:
        origin_label = example['label_des']

        # Normalize some of the labels, eg. English -> Chinese
        if origin_label in label_normalize_dict:
            example['label_des'] = label_normalize_dict[origin_label]
        else:
            # Note: Ideal way is drop these examples
            # which maybe need to change MapDataset
            # Now hard code may hurt performance of `iflytek` dataset
            example['label_des'] = "旅游"

        example["text_label"] = example["label_des"]

        if pattern_id == 0:
            example["sentence1"] = u'作为一款<unk>应用，' + example["sentence"]
        elif pattern_id == 1:
            example["sentence1"] = u'这是一款<unk>应用！' + example["sentence"]
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'。 和<unk>有关'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>!'

        return example


def transform_pet_tnews(example,
                    label_normalize_dict=None,
                    is_test=False,
                    pattern_id=0):
    if is_test:
        example["label_length"] = 2

        if pattern_id == 0:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>的内容！'
        elif pattern_id == 1:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>！'
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'。 包含了<unk>的内容'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'。 综合来讲是<unk>的内容！'

        return example
    else:
        origin_label = example['label_desc']
        # Normalize some of the labels, eg. English -> Chinese
        try:
            example['label_desc'] = label_normalize_dict[origin_label]
        except:
            pass

        if pattern_id == 0:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>的内容！'
        elif pattern_id == 1:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>！'
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'。 包含了<unk>的内容'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'。 综合来讲是<unk>的内容！'

        example["text_label"] = example["label_desc"]

        return example


def transform_pet_eprstmt(example,
                      label_normalize_dict=None,
                      is_test=False,
                      pattern_id=0):
    if is_test:
        example["label_length"] = 1

        if pattern_id == 0:
            example["sentence1"] = u'感觉很<unk>！' + example["sentence"]
        elif pattern_id == 1:
            example["sentence1"] = u'综合来讲很<unk>！，' + example["sentence"]
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'感觉非常<unk>'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'， 我感到非常<unk>'

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        if pattern_id == 0:
            example["sentence1"] = u'感觉很<unk>！' + example["sentence"]
        elif pattern_id == 1:
            example["sentence1"] = u'综合来讲很<unk>！，' + example["sentence"]
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'感觉非常<unk>'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'， 我感到非常<unk>'

        return example


def transform_pet_ocnli(example,
                    label_normalize_dict=None,
                    is_test=False,
                    pattern_id=0):
    if is_test:
        example["label_length"] = 2
        if pattern_id == 0:
            example['sentence1'] = example['sentence1'] + "， <unk>"
        elif pattern_id == 1:
            example["sentence2"] = "和" + example['sentence2'] + u"？看来<unk>一句话"
        elif pattern_id == 2:
            example["sentence1"] = "和" + example['sentence2'] + u"？<unk>一样"
        elif pattern_id == 3:
            example["sentence2"] = "和" + example['sentence2'] + u"？<unk>一句话"

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]
        if pattern_id == 0:
            example['sentence1'] = example['sentence1'] + "， <unk>"
        elif pattern_id == 1:
            example["sentence2"] = "和" + example['sentence2'] + u"？看来<unk>一句话"
        elif pattern_id == 2:
            example["sentence1"] = "和" + example['sentence2'] + u"？<unk>一样"
        elif pattern_id == 3:
            example["sentence2"] = "和" + example['sentence2'] + u"？<unk>一句话"

        return example


def transform_pet_csl(example,
                  label_normalize_dict=None,
                  is_test=False,
                  pattern_id=0):
    if is_test:
        example["label_length"] = 1

        if pattern_id == 0:
            example["sentence1"] = u"本文的关键词<unk>是:" + "，".join(example[
                "keyword"]) + example["abst"]
        elif pattern_id == 1:
            example["sentence1"] = example[
                "abst"] + u"。本文的关键词<unk>是:" + "，".join(example["keyword"])
        elif pattern_id == 2:
            example["sentence1"] = u"本文的内容<unk>是:" + "，".join(example[
                "keyword"]) + example["abst"]
        elif pattern_id == 3:
            example["sentence1"] = example[
                "abst"] + u"。本文的内容<unk>是:" + "，".join(example["keyword"])

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        if pattern_id == 0:
            example["sentence1"] = u"本文的关键词<unk>是:" + "，".join(example[
                "keyword"]) + example["abst"]
        elif pattern_id == 1:
            example["sentence1"] = example[
                "abst"] + u"。本文的关键词<unk>是:" + "，".join(example["keyword"])
        elif pattern_id == 2:
            example["sentence1"] = u"本文的内容<unk>是:" + "，".join(example[
                "keyword"]) + example["abst"]
        elif pattern_id == 3:
            example["sentence1"] = example[
                "abst"] + u"。本文的内容<unk>是:" + "，".join(example["keyword"])

        return example


def transform_pet_csldcp(example,
                     label_normalize_dict=None,
                     is_test=False,
                     pattern_id=0):
    if is_test:
        example["label_length"] = 2

        if pattern_id == 0:
            example["sentence1"] = u'这篇关于<unk>的文章讲了' + example["content"]
        elif pattern_id == 1:
            example["sentence1"] = example["content"] + u'和<unk>息息相关'
        elif pattern_id == 2:
            example["sentence1"] = u'这是一篇和<unk>息息相关的文章' + example["content"]
        elif pattern_id == 3:
            example["sentence1"] = u'很多很多<unk>的文章！' + example["content"]

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        normalized_label = label_normalize_dict[origin_label]
        example['text_label'] = normalized_label
        if pattern_id == 0:
            example["sentence1"] = u'这篇关于<unk>的文章讲了' + example["content"]
        elif pattern_id == 1:
            example["sentence1"] = example["content"] + u'和<unk>息息相关'
        elif pattern_id == 2:
            example["sentence1"] = u'这是一篇和<unk>息息相关的文章' + example["content"]
        elif pattern_id == 3:
            example["sentence1"] = u'很多很多<unk>的文章！' + example["content"]

        return example


def transform_pet_bustm(example,
                    label_normalize_dict=None,
                    is_test=False,
                    pattern_id=0):
    if is_test:
        # Label: ["很"， "不"]
        example["label_length"] = 1
        if pattern_id == 0:
            example['sentence1'] = "<unk>是一句话. " + example['sentence1'] + "，"
        elif pattern_id == 1:
            example['sentence2'] = "，" + example['sentence2'] + "。<unk>是一句话. "
        elif pattern_id == 2:
            example['sentence1'] = "讲的<unk>是一句话。" + example['sentence1'] + "，"
        elif pattern_id == 3:
            example['sentence1'] = "，" + example['sentence2'] + "。讲的<unk>是一句话. "

        return example
    else:
        origin_label = str(example["label"])

        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]
        if pattern_id == 0:
            example['sentence1'] = "<unk>是一句话. " + example['sentence1'] + "，"
        elif pattern_id == 1:
            example['sentence2'] = "，" + example['sentence2'] + "。<unk>是一句话. "
        elif pattern_id == 2:
            example['sentence1'] = "讲的<unk>是一句话。" + example['sentence1'] + "，"
        elif pattern_id == 3:
            example['sentence1'] = "，" + example['sentence2'] + "。讲的<unk>是一句话. "

        return example


def transform_pet_chid(example,
                   label_normalize_dict=None,
                   is_test=False,
                   pattern_id=0):

    if is_test:
        example["label_length"] = 4
        example["sentence1"] = example["content"].replace("#idiom#", "淠")

        return example
    else:
        label_index = int(example['answer'])
        candidates = example["candidates"]
        example["text_label"] = candidates[label_index]

        # Note: `#idom#` represent a idom which must be replaced with rarely-used Chinese characters
        # to get the label's position after the text processed by tokenizer
        #ernie
        example["sentence1"] = example["content"].replace("#idiom#", "淠")

        return example


def transform_pet_cluewsc(example,
                      label_normalize_dict=None,
                      is_test=False,
                      pattern_id=0):
    if is_test:
        example["label_length"] = 2
        text = example["text"]
        span1_text = example["target"]["span1_text"]
        span2_text = example["target"]["span2_text"]

        # example["sentence1"] = text.replace(span2_text, span1_text)
        if pattern_id == 0:
            example["sentence1"] = text + span2_text + "<unk>地指代" + span1_text
        elif pattern_id == 1:
            example["sentence1"] = text + span2_text + "<unk>地意味着" + span1_text
        elif pattern_id == 2:
            example["sentence1"] = text + span2_text + "<unk>地代表" + span1_text
        elif pattern_id == 3:
            example["sentence1"] = text + span2_text + "<unk>地表示了" + span1_text

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]
        # example['text_label'] = origin_label
        text = example["text"]
        span1_text = example["target"]["span1_text"]
        span2_text = example["target"]["span2_text"]

        # example["sentence1"] = text.replace(span2_text, span1_text)
        if pattern_id == 0:
            example["sentence1"] = text + span2_text + "<unk>地指代" + span1_text
        elif pattern_id == 1:
            example["sentence1"] = text + span2_text + "<unk>地意味着" + span1_text
        elif pattern_id == 2:
            example["sentence1"] = text + span2_text + "<unk>地代表" + span1_text
        elif pattern_id == 3:
            example["sentence1"] = text + span2_text + "<unk>地表示了" + span1_text


        return example


def clue_pt_preprocess(example,
                       args,
                       task_name,
                       p_embedding_num=2,
                       is_test=False):
    """
    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        p_embedding_num(obj:`int`) The number of p-embedding.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
        mask_positions(obj: `list[int]`): The list of mask_positions.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
    """
    example = transform_pt_fn_dict[task_name](example, label_normalize_dict=args.label_dict)
    tokenizer = args.tokenizer
    max_seq_length = args.max_seq_length

    # Insert "[MASK]" after "[CLS]"
    start_mask_position = 1
    sentence1 = example["sentence1"]

    encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
    src_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    # Step1: gen mask ids
    if is_test:
        label_length = example["label_length"]
    else:
        text_label = example["text_label"]
        label_length = len(text_label)

    mask_tokens = ["[MASK]"] * label_length
    mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)

    # Step2: gen p_token_ids
    p_tokens = ["[unused{}]".format(i) for i in range(p_embedding_num)]
    p_token_ids = tokenizer.convert_tokens_to_ids(p_tokens)

    # Step3: Insert "[MASK]" to src_ids based on start_mask_position
    src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
                                                          start_mask_position:]

    # Stpe4: Insert P-tokens at begin of sentence
    src_ids = p_token_ids + src_ids

    # calculate mask_positions
    mask_positions = [
        index + start_mask_position + len(p_token_ids)
        for index in range(label_length)
    ]

    if "sentence2" in example:
        encoded_inputs = tokenizer(
            text=example["sentence2"], max_seq_len=max_seq_length)
        sentence2_src_ids = encoded_inputs["input_ids"][1:]
        src_ids += sentence2_src_ids
        token_type_ids += [1] * len(sentence2_src_ids)

    token_type_ids = [0] * len(src_ids)

    assert len(src_ids) == len(
        token_type_ids), "length src_ids, token_type_ids must be equal"

    if is_test:
        return src_ids, token_type_ids, mask_positions
    else:
        mask_lm_labels = tokenizer(
            text=text_label, max_seq_len=max_seq_length)["input_ids"][1:-1]

        assert len(mask_lm_labels) == len(
            mask_positions
        ) == label_length, "length of mask_lm_labels:{} mask_positions:{} label_length:{} not equal".format(
            mask_lm_labels, mask_positions, text_label)
        return src_ids, token_type_ids, mask_positions, mask_lm_labels


def chid_pt_preprocess(example,
                       args,
                       task_name,
                       p_embedding_num=2,
                       is_test=False):
    """
    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        p_embedding_num(obj:`int`) The number of p-embedding.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
        mask_positions(obj: `list[int]`): The list of mask_positions.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
    """

    example = transform_pt_fn_dict[task_name](example, label_normalize_dict=args.label_dict)
    tokenizer = args.tokenizer
    max_seq_length = args.max_seq_length
    # FewClue Task `Chid`' label's position must be calculated by special token: "淠"
    seg_tokens = tokenizer.tokenize(example["sentence1"])

    # find insert position of `[MASK]`
    start_mask_position = seg_tokens.index("淠") + 1
    seg_tokens.remove("淠")
    sentence1 = "".join(seg_tokens)
    candidates = example["candidates"]
    candidate_labels_ids = [
        tokenizer(text=idom)["input_ids"][1:-1] for idom in candidates
    ]

    sentence1 = example["sentence1"]

    encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
    src_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    # Step1: gen mask ids
    if is_test:
        label_length = example["label_length"]
    else:
        text_label = example["text_label"]
        label_length = len(text_label)

    mask_tokens = ["[MASK]"] * label_length
    mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)

    # Step2: gen p_token_ids
    p_tokens = ["[unused{}]".format(i) for i in range(p_embedding_num)]
    p_token_ids = tokenizer.convert_tokens_to_ids(p_tokens)

    # Step3: Insert "[MASK]" to src_ids based on start_mask_position
    src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
                                                          start_mask_position:]

    # Stpe4: Insert P-tokens at begin of sentence
    src_ids = p_token_ids + src_ids

    # calculate mask_positions
    mask_positions = [
        index + start_mask_position + len(p_token_ids)
        for index in range(label_length)
    ]

    token_type_ids = [0] * len(src_ids)

    assert len(src_ids) == len(
        token_type_ids), "length src_ids, token_type_ids must be equal"

    if is_test:
        return src_ids, token_type_ids, mask_positions, candidate_labels_ids
    else:
        mask_lm_labels = tokenizer(
            text=text_label, max_seq_len=max_seq_length)["input_ids"][1:-1]

        assert len(mask_lm_labels) == len(
            mask_positions
        ) == label_length, "length of mask_lm_labels:{} mask_positions:{} label_length:{} not equal".format(
            mask_lm_labels, mask_positions, text_label)

        return src_ids, token_type_ids, mask_positions, mask_lm_labels, candidate_labels_ids


def transform_pt_iflytek(example, label_normalize_dict=None, is_test=False):
    if is_test:
        # When do_test, set label_length field to point
        # where to insert [MASK] id
        example["label_length"] = 2
        example["sentence1"] = example["sentence"]

        return example
    else:
        origin_label = example['label_des']

        # Normalize some of the labels, eg. English -> Chinese
        if origin_label in label_normalize_dict:
            example['label_des'] = label_normalize_dict[origin_label]
        else:
            # Note: Ideal way is drop these examples
            # which maybe need to change MapDataset
            # Now hard code may hurt performance of `iflytek` dataset
            example['label_des'] = "旅游"

        example["text_label"] = example["label_des"]
        example["sentence1"] = example["sentence"]

        return example


def transform_pt_tnews(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 2
        example["sentence1"] = example["sentence"]

        return example
    else:
        origin_label = example['label_desc']
        # Normalize some of the labels, eg. English -> Chinese
        try:
            example['label_desc'] = label_normalize_dict[origin_label]
        except:
            pass
        example["sentence1"] = example["sentence"]
        example["text_label"] = example["label_desc"]

        return example


def transform_pt_eprstmt(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 1
        example['sentence1'] = example["sentence"]
        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]
        example['sentence1'] = example["sentence"]

        return example


def transform_pt_ocnli(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 1
        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        return example


def transform_pt_csl(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 1
        example["sentence1"] = "本文的关键词是:" + "，".join(example[
                                                         "keyword"]) + example["abst"]

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        example["sentence1"] = "本文的关键词是:" + "，".join(example[
                                                         "keyword"]) + example["abst"]

        return example


def transform_pt_csldcp(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 2
        example["sentence1"] = example["content"]

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        normalized_label = label_normalize_dict[origin_label]
        example['text_label'] = normalized_label
        example["sentence1"] = example["content"]

        return example


def transform_pt_bustm(example, label_normalize_dict=None, is_test=False):
    if is_test:
        # Label: ["很"， "不"]
        example["label_length"] = 1
        return example
    else:
        origin_label = str(example["label"])

        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        return example


def transform_pt_chid(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 4
        example["sentence1"] = example["content"].replace("#idiom#", "淠")
        del example["content"]

        return example
    else:

        label_index = int(example['answer'])
        candidates = example["candidates"]
        example["text_label"] = candidates[label_index]

        # Note: `#idom#` represent a idom which must be replaced with rarely-used Chinese characters
        # to get the label's position after the text processed by tokenizer

        example["sentence1"] = example["content"].replace("#idiom#", "淠")

        return example


def transform_pt_cluewsc(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 2
        text = example["text"]
        span1_text = example["target"]["span1_text"]
        span2_text = example["target"]["span2_text"]
        example["sentence1"] = text + span2_text + "指代" + span1_text

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        text = example["text"]
        span1_text = example["target"]["span1_text"]
        span2_text = example["target"]["span2_text"]
        example["sentence1"] = text + span2_text + "指代" + span1_text

        return example


transform_pt_fn_dict = {
    "iflytek": transform_pt_iflytek,
    "tnews": transform_pt_tnews,
    "eprstmt": transform_pt_eprstmt,
    "bustm": transform_pt_bustm,
    "ocnli": transform_pt_ocnli,
    "csl": transform_pt_csl,
    "csldcp": transform_pt_csldcp,
    "cluewsc": transform_pt_cluewsc,
    "chid": transform_pt_chid
}


transform_pet_fn_dict = {
    "iflytek": transform_pet_iflytek,
    "tnews": transform_pet_tnews,
    "eprstmt": transform_pet_eprstmt,
    "bustm": transform_pet_bustm,
    "ocnli": transform_pet_ocnli,
    "csl": transform_pet_csl,
    "csldcp": transform_pet_csldcp,
    "cluewsc": transform_pet_cluewsc,
    "chid": transform_pet_chid
}
