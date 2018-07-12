import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """ Embedding layer before encoder """

    def __init__(self, vocab_size, embedding_dim):
        super(InputEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)


class PositionalEncoding():
    """ PE represents position of tokens and is added to InputEmbedding.
    PE has the same dimension with InputEmbedding. There are many choices of
    PE, but we use sine and cosine functions as decribed in paper.
    """
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def forward(self, sentence):
        """
        input:
            maybe input: [batch(length of sentence), embedding_dim]
            tensor([[[-0.0251, -1.6902,  0.7172],
                     [-0.6431,  0.0748,  0.6969],
                     [ 1.4970,  1.3448, -0.9685],
                     [-0.3677, -2.7265, -0.1685]],

                    [[ 1.4970,  1.3448, -0.9685],
                     [ 0.4362, -0.4004,  0.9400],
                     [-0.6431,  0.0748,  0.6969],
                     [ 0.9124, -2.3616,  1.1151]]])
        """
        return inputs


    def posEnc(pos, i):
        """ pos is the position and i is the dimension"""
        return math.sin(pos/math.pow(10000, (2*i/self.embedding_dim)))


def pre_encoder_test():
    embedding_dim = 128
    vocab = Vocabulary()
    vocab.build("data/corpus.tc.en")
    embed_model = InputEmbedding(16000, embedding_dim)
    pos_enc_model = PositionalEncoding(embedding_dim)
    with open("data/corpus.tc.en", "r") as f:
        for line in f.readlines():
            word_idxs = vocab.setence2vector(line.strip())
            v = autograd.Variable(torch.LongTensor(word_idxs))
            # TODO idk...
            result = embed_model(v) + pos_enc_model(v)
    print(model)


pre_encoder_test()

