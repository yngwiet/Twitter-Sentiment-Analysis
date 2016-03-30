#!/usr/bin/python

import pickle
import os
import random
from preproc_fea_extraction import Preprocessor, FeatureExtractor
import nltk
from nltk import MaxentClassifier
import svm
from svmutil import *


def main():
    # if preprocessed data was stored previously, just load it
    if os.path.isfile('./data/processed/preptrainingdata.pickle') \
            and os.path.isfile('./data/processed/preptestdata.pickle'):
        preptrainingdata_f = open('./data/processed/preptrainingdata.pickle', 'r')
        preptrainingdata = pickle.load(preptrainingdata_f)

        preptestdata_f = open('./data/processed/preptestdata.pickle', 'r')
        preptestdata = pickle.load(preptestdata_f)

        preptrainingdata_f.close()
        preptestdata_f.close()

    else:
        # preprocess training and test data and store them
        trainingdatapath = './data/original/origintrainingdata.csv'
        testdatapath = './data/original/origintestdata.csv'

        preprocessor = Preprocessor(trainingdatapath, testdatapath)

        [training, test] = preprocessor.read_data(2000, 2000)

        # preprocessing step
        for row in training+test:
            row[0] = preprocessor.preprocess(row[0])

        preptrainingdata = training
        preptestdata = test

        # store preprocessed training data
        save_documents = open('./data/processed/preptrainingdata.pickle', 'w')
        pickle.dump(preptrainingdata, save_documents)
        save_documents.close()

        # store preprocessed test data
        save_documents = open('./data/processed/preptestdata.pickle', 'w')
        pickle.dump(preptestdata, save_documents)
        save_documents.close()

    if os.path.isfile('./data/processed/trainingfeaset.pickle') \
            and os.path.isfile('./data/processed/testfeaset.pickle')\
            and os.path.isfile('./data/processed/word_features.pickle'):

        trainingfeaset_f = open('./data/processed/trainingfeaset.pickle', 'r')
        trainingfeaset = pickle.load(trainingfeaset_f)

        testfeaset_f = open('./data/processed/testfeaset.pickle', 'r')
        testfeaset = pickle.load(testfeaset_f)

        word_features_f = open('./data/processed/word_features.pickle', 'r')
        word_features = pickle.load(word_features_f)

        trainingfeaset_f.close()
        testfeaset_f.close()
        word_features_f.close()

    else:
        # feature extraction and feature set construction and store them
        fea_extractor = FeatureExtractor()
        all_words = []

        for row in preptrainingdata+preptestdata:
            all_words.extend(fea_extractor.getfeavector(row[0]))

        word_features = fea_extractor.getfeatures(all_words, 4000)

        del all_words  # release some memory

        trainingfeaset = [(fea_extractor.construct_feaset(row[0], word_features), row[1]) for row in preptrainingdata]
        testfeaset = [(fea_extractor.construct_feaset(row[0], word_features), row[1]) for row in preptestdata]

        # random.shuffle(trainingfeaset)
        # random.shuffle(testfeaset)

        save_documents = open('./data/processed/word_features.pickle', 'w')
        pickle.dump(word_features, save_documents)
        save_documents.close()

        save_documents = open('./data/processed/trainingfeaset.pickle', 'w')
        pickle.dump(trainingfeaset, save_documents)
        save_documents.close()

        save_documents = open('./data/processed/testfeaset.pickle', 'w')
        pickle.dump(testfeaset, save_documents)
        save_documents.close()

    # Naive Bayes
    if os.path.isfile('./data/processed/NB_classifier.pickle'):
        NB_classifier_f = open("./data/processed/NB_classifier.pickle", "r")
        NB_classifier = pickle.load(NB_classifier_f)
        NB_classifier_f.close()

    else:
        NB_classifier = nltk.NaiveBayesClassifier.train(trainingfeaset)
        save_classifier = open("./data/processed/NB_classifier.pickle", "w")
        pickle.dump(NB_classifier, save_classifier)
        save_classifier.close()

    print("Naive Bayes Classifier accuracy percent:", (nltk.classify.accuracy(NB_classifier, testfeaset)) * 100)
    print NB_classifier.show_most_informative_features(10)

    # Maximum Entropy
    if os.path.isfile('./data/processed/MaxEntClassifier.pickle'):
        MaxEntClassifier_f = open('./data/processed/MaxEntClassifier.pickle','r')
        MaxEntClassifier = pickle.load(MaxEntClassifier_f)
        MaxEntClassifier_f.close()

    else:
        MaxEntClassifier = MaxentClassifier.train(trainingfeaset, algorithm='GIS', max_iter=10)
        save_classifier = open("./data/processed/MaxEntClassifier2.pickle", "w")
        pickle.dump(MaxEntClassifier, save_classifier)
        save_classifier.close()

    print "MaxEnt Classifier accuracy percent:", nltk.classify.accuracy(MaxEntClassifier, testfeaset)
    print MaxEntClassifier.show_most_informative_features(10)

    fea_extractor = FeatureExtractor()
    trainingset = fea_extractor.construct_svm_feaset(preptrainingdata, word_features)
    problem = svm_problem(trainingset['labels'], trainingset['feature_vectors'])
    param = svm_parameter('-q')
    param.kernel_type = LINEAR
    svm_classifier = svm_train(problem, param)
    svm_save_model('./data/svm_classifier', svm_classifier)

    testset = fea_extractor.construct_svm_feaset(preptestdata, word_features)
    p_labels, p_accs, p_vals = svm_predict(testset['labels'], testset['feature_vectors'], svm_classifier)

    print p_labels
    print p_accs

if __name__ == "__main__":
    main()
