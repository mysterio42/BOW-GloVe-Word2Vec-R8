import argparse

from utils.config import embeds
from utils.data import load_data
from utils.model import train_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding', choices=embeds.keys(), type=str, required=True,
                        help='Glove Embedding or Word2Vec Embedding')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = load_data(train_path='data/r8-train-all-terms.txt',
                     test_path='data/r8-test-all-terms.txt')

    param = embeds[args.embedding]
    embedding = param['model'](param['path'])
    train_model(embedding, data)
