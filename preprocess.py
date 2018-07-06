
def build_vocab(path):

    word_to_idx, idx_to_word = {}, []
    index = 0

    f = open(path, "r")

    for line in f.readlines():
        word = line.strip()
        word_to_idx[word] = index
        idx_to_word.append(word)
        index += 1


def sentence_to_vector(sentence):
    idxs = [word_to_idx[w] for x in sentence.split(" ")]

