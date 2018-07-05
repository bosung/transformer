import argparse


def data_loader(args):
    train_src = args.train_src


def train(args):
    print(args.vocab)
    print(args.train_src)
    print(args.train_trg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab', required=True, help='vocabulary file')
    parser.add_argument('-word_dim', default=300)
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_trg', required=True)

    args = parser.parse_args()

    train(args)

