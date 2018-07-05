
class InputEmbedding(nn.Module):
    """ Embedding layer before encoder """

    def __init__(self, vocab_size, embedding_dim):
        super(InputEmbedding).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # TODO add positional embedding
        #
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        return embeds

