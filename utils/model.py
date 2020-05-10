from sklearn.ensemble import RandomForestClassifier


def train_model(embedding, data):
    embed_features_train = embedding.transform(data['train']['features'])
    embed_test_train = embedding.transform(data['test']['features'])

    model = RandomForestClassifier(n_estimators=200)
    model.fit(embed_features_train, data['train']['labels'])

    print(f" train score {model.score(embed_features_train, data['train']['labels'])} ")
    print(f" test score {model.score(embed_test_train, data['test']['labels'])} ")
