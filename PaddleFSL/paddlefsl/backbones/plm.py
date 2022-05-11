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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers.ernie.modeling import ErniePretrainingCriterion, ErniePretrainedModel, ErniePretrainingHeads
from paddlenlp.transformers.bert.modeling import BertPretrainedModel, BertPretrainingHeads
from paddlenlp.transformers.albert.modeling import AlbertPretrainedModel, AlbertMLMHead, AlbertForPretraining


class ErnieForPretraining(ErniePretrainedModel):
    def __init__(self, ernie):
        super(ErnieForPretraining, self).__init__()
        self.ernie = ernie
        self.cls = ErniePretrainingHeads(
            self.ernie.config["hidden_size"],
            self.ernie.config["vocab_size"],
            self.ernie.config["hidden_act"],
            embedding_weights=self.ernie.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        with paddle.static.amp.fp16_guard():
            outputs = self.ernie(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask)

            sequence_output, pooled_output = outputs[:2]

            max_len = input_ids.shape[1]
            new_masked_positions = []
            # masked_positions: [bs, label_length]
            for bs_index, mask_pos in enumerate(masked_positions.numpy()):
                for pos in mask_pos:
                    new_masked_positions.append(bs_index * max_len + pos)

            # new_masked_positions: [bs * label_length, 1]
            new_masked_positions = np.array(new_masked_positions).astype(
                'int32')
            new_masked_positions = paddle.to_tensor(new_masked_positions)

            sequence_output = sequence_output.reshape([-1, sequence_output.shape[-1]])
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, new_masked_positions)
            return prediction_scores

    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):

        prediction_logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            masked_positions=masked_positions)

        softmax_fn = paddle.nn.Softmax()
        return softmax_fn(prediction_logits)


class BertForPretraining(BertPretrainedModel):
    def __init__(self, bert):
        super(BertForPretraining, self).__init__()
        self.bert = bert
        self.cls = BertPretrainingHeads(
            self.bert.config["hidden_size"],
            self.bert.config["vocab_size"],
            self.bert.config["hidden_act"],
            embedding_weights=self.bert.embeddings.word_embeddings.weight)

        self.init_weights(self.bert)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        with paddle.static.amp.fp16_guard():
            outputs = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask)

            sequence_output, pooled_output = outputs[:2]

            max_len = input_ids.shape[1]
            new_masked_positions = []
            # masked_positions: [bs, label_length]
            for bs_index, mask_pos in enumerate(masked_positions.numpy()):
                for pos in mask_pos:
                    new_masked_positions.append(bs_index * max_len + pos)

            # new_masked_positions: [bs * label_length, 1]
            new_masked_positions = np.array(new_masked_positions).astype(
                'int32')
            new_masked_positions = paddle.to_tensor(new_masked_positions)

            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, new_masked_positions)
            return prediction_scores

    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):

        prediction_logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            masked_positions=masked_positions)

        softmax_fn = paddle.nn.Softmax()
        return softmax_fn(prediction_logits)


class AlbertForPretraining(AlbertPretrainedModel):
    def __init__(self, albert):
        super(AlbertForPretraining, self).__init__()
        self.transformer = albert
        self.cls = AlbertMLMHead(
            self.transformer.config['embedding_size'],
            self.transformer.config["vocab_size"],
            self.transformer.config['hidden_size'],
            self.transformer.config['hidden_act'],
        )
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        with paddle.static.amp.fp16_guard():
            outputs = self.transformer(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask)

            sequence_output, pooled_output = outputs[:2]

            max_len = input_ids.shape[1]
            new_masked_positions = []
            # masked_positions: [bs, label_length]
            for bs_index, mask_pos in enumerate(masked_positions.numpy()):
                for pos in mask_pos:
                    new_masked_positions.append(bs_index * max_len + pos)

            # new_masked_positions: [bs * label_length, 1]
            new_masked_positions = np.array(new_masked_positions).astype(
                'int32')
            new_masked_positions = paddle.to_tensor(new_masked_positions)

            sequence_output = paddle.tensor.gather(sequence_output,
                                                   new_masked_positions)
            prediction_scores = self.cls(sequence_output)

            return prediction_scores

    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):

        prediction_logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            masked_positions=masked_positions)

        softmax_fn = paddle.nn.Softmax()
        return softmax_fn(prediction_logits)


class ErnieMLMCriterion(paddle.nn.Layer):
    def __init__(self):
        super(ErnieMLMCriterion, self).__init__()

    def forward(self, prediction_scores, masked_lm_labels, masked_lm_scale=1.0, weights=None):

        masked_lm_labels = paddle.reshape(masked_lm_labels, shape=[-1, 1])
        with paddle.static.amp.fp16_guard():
            if weights is not None:
                masked_lm_loss = F.cross_entropy(
                    input=prediction_scores, label=masked_lm_labels, reduction='none')
                masked_lm_labels *= weights
                masked_lm_loss = masked_lm_loss.mean()
            else:
                masked_lm_loss = F.cross_entropy(
                    input=prediction_scores, label=masked_lm_labels)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            return masked_lm_loss
