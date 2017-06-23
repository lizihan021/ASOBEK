import numpy as np

class SciKitClassifier:
    def __init__(self, train_database, feature_gen, classifier):
        self._classifier = classifier
        self._feature_gen = feature_gen
        inputs = []
        outputs = []
        for data_row in train_database:
            # when the data is debatable simply skip
            if data_row.is_not_valid():
                continue
            inputs.append(np.array(feature_gen(data_row)))
            outputs.append(1 if data_row.is_paraphrase() else 0)
            if (data_row.topic_id % 1000) == 0:
                print("dealing with row " + str(data_row.topic_id), end="\r")
        X, Y = np.array(inputs, dtype=np.float32), np.array(outputs)
#        print(X.shape, Y.shape, outputs)
        self._classifier.fit(X, Y)

    def classify(self, data_row):
        features = self._feature_gen(data_row)
        p = self._classifier.predict(np.array([features], dtype=np.float32))
        return p[0] > 0.5
