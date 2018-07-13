import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from preprocess import Vocabulary

class InputEmbedding(nn.Module):
    """ Embedding layer before encoder """

    def __init__(self, vocab_size, embedding_dim):
        super(InputEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)


class PositionalEncoding(nn.Module):
    """ PE represents position of tokens and is added to InputEmbedding.
    PE has the same dimension with InputEmbedding. There are many choices of
    PE, but we use sine and cosine functions as described in paper.
    """
    def __init__(self, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, inputs):
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
        for pos, sentence in enumerate(inputs):
            sentence = sentence + autograd.Variable(self.posEnc(pos), requires_grad=False)
        return sentence


    def posEnc(self, pos):
        """ pos is the position and i is the dimension"""
        pos_enc = np.array(
                [pos/math.pow(10000, (2*i/self.embedding_dim)) for i in range(self.embedding_dim)])
        pos_enc[0::2] = np.sin(pos_enc[0::2])
        pos_enc[1::2] = np.cos(pos_enc[1::2])
        return torch.FloatTensor(pos_enc)


def pre_encoder_test():
    embedding_dim = 128
    vocab = Vocabulary()
    vocab.load("data/vocab.bpe.32000")
    embed_model = InputEmbedding(32000, embedding_dim)
    pos_enc_model = PositionalEncoding(embedding_dim)

    #sentence = "Res@@ um@@ ption of the session"
    #inputs = vocab.sentence2vector(sentence)
    #inputs = autograd.Variable(torch.LongTensor(inputs))
    #print(inputs)
    #inputs = embed_model(inputs)
    #pos_enc_model(inputs)

    with open("data/train.tok.clean.bpe.32000.en", "r") as f:
        idx = 0
        for line in f.readlines():
            word_idxs = vocab.sentence2vector(line.strip())
            v = autograd.Variable(torch.LongTensor(word_idxs))
            result = embed_model(v)
            result = pos_enc_model(result)
            print(result.view(1, -1))
            idx += 1
            if idx > 10:
                break


if __name__ == "__main__":
    pre_encoder_test()


