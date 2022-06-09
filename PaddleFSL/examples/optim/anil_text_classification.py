"""ANIL example for optimization"""
from __future__ import annotations
from functools import partial
import paddle
from paddle import nn
from paddle.optimizer import Adam
import paddlefsl
from paddlefsl.metaopt.anil import ANILLearner
from paddlenlp.transformers.tokenizer_utils_base import PaddingStrategy
from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.transformers.ernie.modeling import ErnieModel
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
from examples.optim.tools.meta_trainer import Config, Trainer


class SequenceClassifier(nn.Layer):
    """Sequence Classifier"""
    def __init__(self, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, output_size)
    
    def forward(self, embedding):
        """handle the main logic"""
        embedding = self.dropout(embedding)
        logits = self.classifier(embedding)
        return logits
        
def vector_initializer(tokens, head_position, tail_position, max_len, tokenizer: PretrainedTokenizer):
    head_tokens = [tokens[index] for index in head_position]
    tail_tokens = [tokens[index] for index in tail_position]

    sentence = ['[CLS]'] + tokens + ['[SEP]'] + head_tokens + ['&'] + tail_tokens + ['[SEP]']
    
    feature = tokenizer.encode(
        ''.join(sentence),
        max_length=max_len,
        padding=PaddingStrategy.MAX_LENGTH,
        return_tensors='pd',
    )
    return feature['input_ids']
    

if __name__ == '__main__':

    config = Config().parse_args(known_only=True)
    config.device = 'cpu'
    tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
    
    train_dataset = paddlefsl.datasets.few_rel.FewRel('train', vector_initializer=partial(vector_initializer, tokenizer=tokenizer))
    train_dataset._data_list = train_dataset._data_list[:100]
    valid_dataset = paddlefsl.datasets.few_rel.FewRel('valid', vector_initializer=partial(vector_initializer, tokenizer=tokenizer))
    valid_dataset._data_list = valid_dataset._data_list[:100]
    test_dataset = paddlefsl.datasets.few_rel.FewRel('valid', vector_initializer=partial(vector_initializer, tokenizer=tokenizer))
    test_dataset._data_list = test_dataset._data_list[:100]

    feature_model, head_layer = ErnieModel.from_pretrained('ernie-1.0'), SequenceClassifier(hidden_size=768, output_size=config.n_way)

    criterion = nn.CrossEntropyLoss()
    learner = ANILLearner(
        feature_model=feature_model,
        head_layer=head_layer,
        learning_rate=config.inner_lr,
    )
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=config.meta_lr, T_max=config.epochs)
    optimizer = Adam(parameters=learner.parameters(), learning_rate=scheduler)
    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        dev_dataset=valid_dataset,
        test_dataset=test_dataset,
        learner=learner,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )
    trainer.train()
