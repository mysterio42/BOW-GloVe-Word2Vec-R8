import pandas as pd


def load_data(train_path, test_path):
    def helper(path):
        data = pd.read_csv(path, header=None, sep='\t')
        data.columns = ['label', 'content']
        features, labels = data['content'], data['label']
        return features, labels

    train_features, train_labels = helper(train_path)
    test_features, test_labels = helper(test_path)
    data = {
        'train': {
            'features': train_features,
            'labels': train_labels
        },
        'test': {
            'features': test_features,
            'labels': test_labels
        }
    }
    return data
