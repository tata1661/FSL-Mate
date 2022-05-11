import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F


class MatchingNet(nn.Layer):
    """
    MatchingNet Module for memory network of MatchingNet.

    Args:
        n_way(int, optional): Number of classes in a task, default 5.
        n_support(int, optional): Number of training samples per class, default 1.
        n_query(int, optional): Number of query points per class, default 16.
        feat_dim(int, optional): feature dimensions of memory networks, default 1024.

    Examples:
        ..code-block:: python
            train_input = paddle.ones(shape=(5 * (16 + 1), 1024), dtype='float32')  # 5: 5 ways
            model = MatchingNet(5, 1, 16, feat_dim=1024)
            print(model(train_input))   # Tensor of shape [80, 5]

    """
    def __init__(self, n_way, n_support, n_query, feat_dim):
        super(MatchingNet, self).__init__()

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.feat_dim = feat_dim
        self.FCE = FullyContextualEmbedding(self.feat_dim)
        self.G_encoder = nn.LSTM(self.feat_dim, self.feat_dim, 1, direction='bidirectional')
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode_training_set(self, S):
        out_G = self.G_encoder(S.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = S + out_G[:, :S.shape[1]] + out_G[:, S.shape[1]:]
        G_norm = paddle.norm(G, p=2, axis=1).unsqueeze(1).expand_as(G)
        G_normalized = G.divide(G_norm + 0.00001)
        return G, G_normalized

    def get_logprobs(self, f, G, G_normalized, Y_S):
        F = self.FCE(f, G)
        F_norm = paddle.norm(F, p=2, axis=1).unsqueeze(1).expand_as(F)
        F_normalized = F.divide(F_norm + 0.00001)
        scores = self.relu(F_normalized.matmul(G_normalized, transpose_y=True)) * 100
        softmax = self.softmax(scores)
        logprobs = (softmax.matmul(Y_S.cast('float32')) + 1e-6).log()
        return logprobs

    def forward(self, z_all):
        z_all = z_all.reshape([self.n_way, self.n_support + self.n_query, -1])
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        z_support = z_support.reshape([self.n_way * self.n_support, -1])
        z_query = z_query.reshape([self.n_way * self.n_query, -1])
        G, G_normalized = self.encode_training_set(z_support)

        y_s = paddle.to_tensor(np.repeat(range(self.n_way), self.n_support))
        Y_S = F.one_hot(y_s, self.n_way)
        f = z_query
        logprobs = self.get_logprobs(f, G, G_normalized, Y_S)
        return logprobs


class FullyContextualEmbedding(nn.Layer):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.softmax = nn.Softmax()
        self.c_0 = paddle.zeros(shape=[1, feat_dim])
        self.feat_dim = feat_dim

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose([0, 1])
        K = G.shape[0]
        for k in range(K):
            logit_a = h.matmul(G_T, transpose_y=True)
            a = self.softmax(logit_a)
            r = a.matmul(G)
            x = paddle.concat((f, r), 1)

            _, (h, c) = self.lstmcell(x, (h, c))
            h = h + f

        return h
