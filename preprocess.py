
class Vocabulary:

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        print("New Vocabulary!")

    def load(self, vocab_path):
        """ load vocabulary dictionary """
        self.idx = 0
        with open(vocab_path, "r") as f:
            for line in f.readlines():
                word = line.strip()
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
        f.close()
        print("[DONE] load vocabulary from %s" % vocab_path)

    def build(self, train_path, vocab_size=16000):
        """ build vocabulary from train data """
        word_count = {}
        line_num = 0
        with open(train_path, "r") as f:
            for line in f.readlines():
                line_num += 1
                if line_num % 100000 == 0:
                    print("[INFO] read %s lines..." % str(line_num))
                words = [x.strip() for x in line.split(" ")]
                for w in words:
                    if w not in word_count:
                        word_count[w] = 1
                    else:
                        word_count[w] += 1
        f.close()
        print("[INFO] total %s lines" % str(len(line_num)))
        print("[INFO] total %s vocabulary" % str(len(word_count)))
        print("[INFO] build %s vocabulary dict from %s" % (str(vocab_size), train_path))
        vocab_idx = 0
        for key, value in reversed(sorted(word_count.items(), key=lambda i: (i[1], i[0]))):
            self.word2idx[key] = vocab_idx
            self.idx2word[vocab_idx] = key
            vocab_idx += 1
            if vocab_idx == vocab_size:
                break
        print("[DONE] build success!")

    def sentence2vector(self, sentence):
        # OOV -> <unk>
        # TODO lowercass??
        vector = []
        for w in sentence.split(" "):
            if w in self.word2idx:
                vector.append(self.word2idx[w])
            else:
                vector.append(-1)
        return vector


if __name__ == "__main__":
    vocab = Vocabulary()
    vocab.build("data/corpus.tc.en")
    result = vocab.sentence2vector("Obama relation new automatic automa")
    print(result)

