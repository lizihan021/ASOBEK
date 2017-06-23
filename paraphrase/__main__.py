from .models import *
from .evaluate import *
from .features import concat
from .ngram import C1, C2, V1, V2
from .skclassifier import SciKitClassifier
from .word2vec import word2vec_features
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn import svm
import sklearn.preprocessing as pre
import csv
import os.path
import pickle


if __name__ == "__main__":
    print('reading database')
    train_database = read_database("./paraphrase/train.csv", TrainDataRow)
    print('train data size: ' + str(len(train_database)))
    test_database  = read_database("./paraphrase/test.csv", TestDataRow)
    print('test data size: ' + str(len(test_database)))
    # TODO: check database valid?

    features = [C1, C2, V1, V2] + \
        [concat(c, v) for c in [C1, C2] for v in [V1, V2]]

    if not os.path.isfile('./paraphrase/NBclassfier.pkl'):
        with open('./paraphrase/NBclassfier.pkl', 'wb') as output:
            w2v_nb = SciKitClassifier(train_database, word2vec_features, GaussianNB())
            pickle.dump(w2v_nb, output, pickle.HIGHEST_PROTOCOL)
            print('classfier trained and saved')
    else:
        with open('./paraphrase/NBclassfier.pkl', 'rb') as input:
            w2v_nb = pickle.load(input)
            print('classfier loaded')

    print("using gaussian NB")
    with open("./paraphrase/output.csv", 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['test_id', 'is_duplicate'])
        for guess in evaluate(w2v_nb, test_database):
            writer.writerow(guess)
            print("writting row " + str(guess[0]), end="\r")

'''
    for features_gen in features:
        svm_pipeline = \
            make_pipeline(pre.StandardScaler(),  AdaBoostRegressor(n_estimators = 256))
        classifier = SciKitClassifier(train_database, features_gen, svm_pipeline)
        features_gen = concat(features_gen, word2vec_features)
        print("features", features_gen.name)
        for guess in evaluate(classifier, test_database):
            print(guess)
'''
