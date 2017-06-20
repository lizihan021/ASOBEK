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


if __name__ == "__main__":
    train_database = read_database("./paraphrase/train.data", TrainDataRow)
    test_database  = read_database("./paraphrase/test.data", TestDataRow)
    features = [C1, C2, V1, V2] + \
        [concat(c, v) for c in [C1, C2] for v in [V1, V2]]

    w2v_nb = SciKitClassifier(train_database, word2vec_features, GaussianNB())
    print("gaussian NB")
    #print(evaluate(w2v_nb, test_database))
    for guess in evaluate(w2v_nb, test_database):
        print(guess)

    for features_gen in features:
        svm_pipeline = \
            make_pipeline(pre.StandardScaler(),  AdaBoostRegressor(n_estimators = 256))
        classifier = SciKitClassifier(train_database, features_gen, svm_pipeline)
        features_gen = concat(features_gen, word2vec_features)
        print("features", features_gen.name)
        #print(evaluate(classifier, test_database))
        for guess in evaluate(classifier, test_database):
            print(guess)

    # print("\n".join(str(x) for x in train_database[0:10]))
