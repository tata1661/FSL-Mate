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

import argparse
import os
import sys
import random
import time
import json
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from paddlefsl.backbones.plm import AlbertForPretraining, ErnieMLMCriterion, BertForPretraining, ErnieForPretraining
from paddlefsl.datasets.fewnlu.nlu_dataloader import create_dataloader, convert_example
from paddlenlp.transformers.albert.modeling import AlbertModel, AlbertMLMHead
from paddlefsl.datasets.fewnlu.data_glue.task_processors import load_examples, PROCESSORS
from paddlefsl.datasets.fewnlu.data_glue.preprocess import MLMPreprocessor
from paddlefsl.datasets.fewnlu.data_clue.preprocess import chid_pt_preprocess, clue_pt_preprocess, chid_pet_preprocess, clue_pet_preprocess

# yapf: disable
parser = argparse.ArgumentParser()
args = parser.parse_args()


# yapf: enable

def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


# Transform the origin label to the predefined label on FewGLUE datasets.
transform_map = {'entailment': 'yes',
                 'contradiction': 'no',
                 'neutral': 'Maybe',
                 'F': 'no',
                 'T': 'yes',
                 '0': 'no',
                 '1': 'yes',
                 'False': 'no',
                 'True': 'yes',
                 'not_entailment': 'no'}


@paddle.no_grad()
def do_evaluate(model, lan, tokenizer, data_loader, label_normalize_dict):
    model.eval()

    total_num = 0
    correct_num = 0
    top2_num = 0
    top3_num = 0

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]
    label_length = len(normed_labels[0]) if lan == 'zh' else len(normed_labels[0].split(' '))
    all_y_true = np.array([])
    all_y_pred = np.array([])

    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, masked_lm_labels = batch

        # [bs * label_length, vocab_size]
        prediction_probs = model.predict(
            input_ids=src_ids,
            token_type_ids=token_type_ids,
            masked_positions=masked_positions)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]

        # prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = paddle.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size]).numpy()

        # [label_num, label_length]
        label_ids = np.array(
            [tokenizer(label)["input_ids"][1:-1] for label in normed_labels])

        y_pred = np.ones(shape=[batch_size, len(label_ids)])

        # Calculate joint distribution of candidate labels
        for index in range(label_length):
            y_pred *= prediction_probs[:, index, label_ids[:, index]]
        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)

        y_true_index = []
        for masked_lm_label in masked_lm_labels.numpy():
            label_text = "".join(
                tokenizer.convert_ids_to_tokens(masked_lm_label.tolist()))
            try:
                label_index = normed_labels.index(label_text)
            except:
                label_text = label_text[0].upper() + label_text[1:]
                label_index = normed_labels.index(label_text)
            y_true_index.append(label_index)

        y_true_index = np.array(y_true_index)
        all_y_true = np.concatenate([all_y_true,y_true_index],axis=0)
        all_y_pred = np.concatenate([all_y_pred,y_pred_index],axis=0)


        total_num += len(y_true_index)
        correct_num += (y_true_index == y_pred_index).sum()

    return 100 * correct_num / total_num, total_num

@paddle.no_grad()
def do_evaluate_chid(model, lan, tokenizer, data_loader, label_normalize_dict):
    """
        FewCLUE `chid` dataset is specical when evaluate: input slots have
        additional `candidate_label_ids`, so need to customize the
        evaluate function.
    """

    model.eval()

    total_num = 0
    correct_num = 0

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    label_length = len(normed_labels[0])

    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_label_ids = batch

        # [bs * label_length, vocab_size]
        prediction_probs = model.predict(
            input_ids=src_ids,
            token_type_ids=token_type_ids,
            masked_positions=masked_positions)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]

        # prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = paddle.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size]).numpy()

        candidate_num = candidate_label_ids.shape[1]

        # [batch_size, candidate_num(7)]
        y_pred = np.ones(shape=[batch_size, candidate_num])

        for label_idx in range(candidate_num):

            # [bathc_size, label_length(4)]
            single_candidate_label_ids = candidate_label_ids[:, label_idx, :]
            # Calculate joint distribution of candidate labels
            for index in range(label_length):
                # [batch_size,]
                slice_word_ids = single_candidate_label_ids[:, index].numpy()

                batch_single_token_prob = []
                for bs_index in range(batch_size):
                    # [1, 1]
                    single_token_prob = prediction_probs[
                        bs_index, index, slice_word_ids[bs_index]]
                    batch_single_token_prob.append(single_token_prob)

                y_pred[:, label_idx] *= np.array(batch_single_token_prob)

        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)

        y_true_index = []

        for index, masked_lm_label in enumerate(masked_lm_labels.numpy()):
            # [cantidate_num, label_length]
            tmp_candidate_label_ids = candidate_label_ids[index, :, :]
            for idx, label_ids in enumerate(tmp_candidate_label_ids.numpy()):
                if np.equal(label_ids, masked_lm_label).all():
                    y_true_index.append(idx)
                    continue

        y_true_index = np.array(y_true_index)

        total_num += len(y_true_index)
        correct_num += (y_true_index == y_pred_index).sum()

    return 100 * correct_num / total_num, total_num


def do_train(model_name,
             lan,
             task_name,
             save_dir,
             learning_rate,
             batch_size,
             data_path,
             max_seq_length = 256,
             init_from_ckpt = None,
             warmup_proportion = 0.0,
             weight_decay = 0,
             epochs = 10,
             device = 'gpu',
             seed = 1000,
             pattern_id = 1
             ):
    args.task_name = task_name
    args.lan = lan
    args.if_pt = False
    args.max_seq_length = max_seq_length
    args.batch_size = batch_size
    args.pattern_id = pattern_id
    args.data_path = data_path

    paddle.set_device(device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(seed)

    args.if_pt = True
    # Defining the function to read dataset
    if args.lan == 'zh':
        # load dataset
        train_ds, public_test_ds, dev_ds = load_dataset(
            "fewclue",
            name=args.task_name,
            splits=("train_0", "test_public", "dev_0"))
    else:
        def read(set_type):
            a = load_examples(args.task_name, args.data_path, set_type)
            return a

        evaluate_fn = do_evaluate
        train_ds = load_dataset(read, set_type='train', lazy=False)
        dev_ds = load_dataset(read, set_type='dev32', lazy=False)
        public_test_ds = load_dataset(read, set_type='dev', lazy=False)

    if args.lan == 'zh':
        label_normalize_json = os.path.join("./label_normalized",
                                            args.task_name + ".json")
        with open(label_normalize_json, 'r', encoding="utf-8") as f:
            label_dict = json.load(f)

        tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(model_name)
        model = ErnieForPretraining.from_pretrained(model_name)
        args.label_dict = label_dict

    else:
        tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(model_name)
        model = BertForPretraining.from_pretrained(model_name)

        processor = PROCESSORS[args.task_name]()
        label_list = processor.get_labels()
        label_dict = {}

        for label in label_list:
            label_dict[label] = transform_map[label]
        args.label_list = label_list

    args.tokenizer = tokenizer
    convert_example_fn = convert_example
    evaluate_fn = do_evaluate if args.task_name != "chid" else do_evaluate_chid

    if args.lan == 'zh':
        if args.task_name == 'chid':
            fn = chid_pt_preprocess if args.if_pt else chid_pet_preprocess
            preprocess = partial(fn, args=args, task_name=args.task_name)

        else:
            fn = clue_pt_preprocess if args.if_pt else clue_pet_preprocess
            preprocess = partial(fn, args=args, task_name=args.task_name)

    else:
        preprocess = MLMPreprocessor(args, args.task_name, pattern_id=args.pattern_id, if_pt=args.if_pt)

    if args.task_name != "chid":
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # masked_lm_labels
        ): [data for data in fn(samples)]
    else:
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_labels_ids]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # masked_lm_labels
            Stack(dtype="int64"),  # candidate_labels_ids [candidate_num, label_length]
        ): [data for data in fn(samples)]

    trans_func = partial(
        convert_example_fn,
        preprocess=preprocess,
        lan=args.lan)

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    public_test_data_loader = create_dataloader(
        public_test_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    # Load the pretrained model
    if init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(init_from_ckpt))

    mlm_loss_fn = ErnieMLMCriterion()

    num_training_steps = len(train_data_loader) * epochs

    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps,
                                         warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    best_test_acc = 0
    global_step = 0
    tic_train = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):

            src_ids = batch[0]
            token_type_ids = batch[1]
            masked_positions = batch[2]
            masked_lm_labels = batch[3]

            prediction_scores = model(
                input_ids=src_ids,
                token_type_ids=token_type_ids,
                masked_positions=masked_positions)

            loss = mlm_loss_fn(prediction_scores, masked_lm_labels, weights=None)
            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        dev_accuracy, total_num, = evaluate_fn(model, args.lan, tokenizer, dev_data_loader,
                                               label_dict)
        print("epoch:{}, dev_accuracy:{:.3f}, total_num:{}".format(
            epoch, dev_accuracy, total_num))
        # test_accuracy, total_num = evaluate_fn(
        #     model, args.lan, tokenizer, public_test_data_loader, label_dict)
        # print("epoch:{}, test_accuracy:{:.3f}, total_num:{}".format(
        #     epoch, test_accuracy, total_num))

        if rank == 0 and dev_accuracy > best_test_acc:
            best_test_acc = dev_accuracy
            save_dir = os.path.join(save_dir, 'model')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_dir)
    return model, tokenizer, public_test_data_loader, label_dict
