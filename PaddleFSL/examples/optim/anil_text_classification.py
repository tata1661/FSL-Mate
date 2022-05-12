"""ANIL example for optimization"""
from __future__ import annotations
import os
import paddle
from paddle import nn
from paddle.optimizer import Adam
import paddlefsl
from paddlefsl.metaopt.anil import ANILLearner
from paddlenlp.transformers.ernie.modeling import ErnieModel
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
from examples.optim.tools.meta_trainer import Config, Trainer, load_datasets


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
        

if __name__ == '__main__':

    config = Config().parse_args(known_only=True)
    config.device = 'gpu'
    
    train_dataset = paddlefsl.datasets.few_rel.FewRel('train')
    valid_dataset = paddlefsl.datasets.few_rel.FewRel('valid')
    test_dataset = paddlefsl.datasets.few_rel.FewRel('valid')

    config.tracking_uri = os.environ.get('TRACKING_URI', None)
    config.experiment_id = os.environ.get('EXPERIMENT_ID', None)

    tokenzier = ErnieTokenizer.from_pretrained('ernie-1.0')
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
        tokenizer=tokenzier
    )
    trainer.train()
