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
from pycorenlp import StanfordCoreNLP
import sklearn.preprocessing as pre
import csv
import os.path
import pickle


if __name__ == "__main__":
    # test = read_sentences('I saw a dog today','A dog was seen by me',TestDataRow)
    print('reading database')
    train_database = read_database("./data/train.csv", TrainDataRow)
    print('train data size: ' + str(len(train_database)))
    test_database  = read_database("./data/test.csv", TestDataRow)
    print('test data size: ' + str(len(test_database)))
    # TODO: check database valid?

    # nlp = StanfordCoreNLP('http://localhost:9000')
    # text = (
    #     'Pusheen and Smitha walked along the beach. '
    #     'Pusheen wanted to surf, but fell off the surfboard.')
    # output = nlp.annotate(text, properties={
    #     'annotators': 'tokenize,ssplit,pos,depparse,parse',
    #     'outputFormat': 'json'
    #     })

    # print(output['sentences'][0]['parse'])

    features = [C1, C2, V1, V2] + \
        [concat(c, v) for c in [C1, C2] for v in [V1, V2]]

    if not os.path.isfile('./data/NBclassfier.pkl'):
        with open('./data/NBclassfier.pkl', 'wb') as output:
            w2v_nb = SciKitClassifier(train_database, word2vec_features, GaussianNB())
            pickle.dump(w2v_nb, output, pickle.HIGHEST_PROTOCOL)
            print('NB classfier trained and saved')
    else:
        with open('./data/NBclassfier.pkl', 'rb') as input:
            w2v_nb = pickle.load(input)
            print('NB classfier loaded')

    print("using gaussian NB")
    with open("./data/output.csv", 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['test_id', 'is_duplicate'])
        for guess in evaluate(w2v_nb, test_database):
            writer.writerow(guess)
            if guess[0] % 1000 == 0:
                print("writting row " + str(guess[0]), end="\r")
    
    for features_gen in features:
        svm_pipeline = \
            make_pipeline(pre.StandardScaler(),  AdaBoostRegressor(n_estimators = 256))

        if not os.path.isfile(('./data/svm'+features_gen.name+'.pkl')):
            with open(('./data/svm'+features_gen.name+'.pkl'), 'wb') as output:
                classifier = SciKitClassifier(train_database, features_gen, svm_pipeline)
                pickle.dump(classifier , output, pickle.HIGHEST_PROTOCOL)
                print('svm' + features_gen.name + ' classfier trained and saved')
        else:
            with open(('./data/svm'+features_gen.name+'.pkl'), 'rb') as input:
                classifier = pickle.load(input)
                print('svm' + features_gen.name + ' classfier loaded')

        with open(('./data/output'+features_gen.name+'.csv'), 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['test_id', 'is_duplicate'])
            for guess in evaluate(classifier, test_database):
                writer.writerow(guess)
                if guess[0] % 1000 == 0:
                    print("writting row " + str(guess[0]), end="\r")

    print('finished')

