#!/usr/bin/python

import pickle
from preproc_fea_extraction import Preprocessor, FeatureExtractor
import nltk
from nltk import MaxentClassifier
from svmutil import *
import time


def main():
    # for feature extraction
    fea_extractor = FeatureExtractor()

    # if preprocessed data was stored previously, just load it
    # for what is mean by "preprocessed", refer to preprocess method in preproc_fea_extraction.py
    if os.path.isfile('./data/processed/preptrainingdata.pickle') \
            and os.path.isfile('./data/processed/preptestdata.pickle'):

        print "preptrainingdata and preptestdata detected, load files..."

        preptrainingdata_f = open('./data/processed/preptrainingdata.pickle', 'r')
        preptrainingdata = pickle.load(preptrainingdata_f)

        preptestdata_f = open('./data/processed/preptestdata.pickle', 'r')
        preptestdata = pickle.load(preptestdata_f)

        preptrainingdata_f.close()
        preptestdata_f.close()

    else:

        print "no preptrainingdata and preptestdata detected, create from scratch..."

        # preprocess training and test data and store them
        trainingdatapath = './data/original/origintrainingdata.csv'
        testdatapath = './data/original/origintestdata.csv'

        preprocessor = Preprocessor(trainingdatapath, testdatapath)

        [training, test] = preprocessor.read_data(10000, 10000)

        print "reading training data and all test data done..."

        print "length of training", len(training)

        # preprocessing step
        for row in training+test:
            row[0] = preprocessor.preprocess(row[0])

        preptrainingdata = training
        preptestdata = test

        print "preprocessing done..."

        # store preprocessed training data
        save_documents = open('./data/processed/preptrainingdata.pickle', 'w')
        pickle.dump(preptrainingdata, save_documents)
        save_documents.close()

        # store preprocessed test data
        save_documents = open('./data/processed/preptestdata.pickle', 'w')
        pickle.dump(preptestdata, save_documents)
        save_documents.close()

    # if training feature set and test feature set are stored previously, just load them
    # these feature set are used by Naive Bayes and Maximum Entropy
    # word_features contains the names of features (which are words)
    # e.g. a word is a feature, feature name is the word, value is True or False
    if os.path.isfile('./data/processed/trainingfeaset.pickle') \
            and os.path.isfile('./data/processed/testfeaset.pickle')\
            and os.path.isfile('./data/processed/word_features.pickle'):

        print "trainingfeaset, testfeaset and word_features detected, load files..."

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

        print "no trainingfeaset, testfeaset and word_features detected, create from scratch..."

        # feature extraction and feature set construction and store them
        all_words = []

        for row in preptrainingdata+preptestdata:
            all_words.extend(fea_extractor.getfeavector(row[0]))

        print "generating all_words done..."
        print "start generating word_features..."

        # set desired # of features in the second parameter
        word_features = fea_extractor.getfeatures(all_words, 5000)

        print "generating word_features done..."

        del all_words  # release some memory

        trainingfeaset = [(fea_extractor.construct_feaset(row[0], word_features), row[1]) for row in preptrainingdata]
        testfeaset = [(fea_extractor.construct_feaset(row[0], word_features), row[1]) for row in preptestdata]

        print "generating trainingfeaset and testfeaset done... great progress!"

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

        print "storing training and test featureset files done..."

    # Naive Bayes
    print "Naive Bayes start..."

    if os.path.isfile('./data/processed/NB_classifier.pickle'):
        NB_classifier_f = open("./data/processed/NB_classifier.pickle", "r")
        NB_classifier = pickle.load(NB_classifier_f)
        NB_classifier_f.close()

    else:
        start = time.time()
        NB_classifier = nltk.NaiveBayesClassifier.train(trainingfeaset)
        NB_trainingtime = time.time() - start

        print "Naive Bayes training time:", NB_trainingtime

        save_classifier = open("./data/processed/NB_classifier.pickle", "w")
        pickle.dump(NB_classifier, save_classifier)
        save_classifier.close()

    print "Naive Bayes Classifier accuracy percent:", (nltk.classify.accuracy(NB_classifier, testfeaset)) * 100
    print NB_classifier.show_most_informative_features(10)

    # Maximum Entropy
    print "Maximum Entropy start..."

    if os.path.isfile('./data/processed/MaxEntClassifier.pickle'):
        MaxEntClassifier_f = open('./data/processed/MaxEntClassifier.pickle','r')
        MaxEntClassifier = pickle.load(MaxEntClassifier_f)
        MaxEntClassifier_f.close()

    else:
        start = time.time()
        MaxEntClassifier = MaxentClassifier.train(trainingfeaset, algorithm='GIS', max_iter=10)
        MaxEnt_trainingtime = time.time() - start

        print "Maximum Entropy training time:", MaxEnt_trainingtime

        save_classifier = open("./data/processed/MaxEntClassifier.pickle", "w")
        pickle.dump(MaxEntClassifier, save_classifier)
        save_classifier.close()

    print "MaxEnt Classifier accuracy percent:", nltk.classify.accuracy(MaxEntClassifier, testfeaset)
    print MaxEntClassifier.show_most_informative_features(10)

    # SVM
    print "SVM start..."

    testset = fea_extractor.construct_svm_feaset(preptestdata, word_features)

    if os.path.isfile('./data/processed/svm_classifier.model'):

        svm_classifier = svm_load_model('./data/processed/svm_classifier.model')

    else:

        trainingset = fea_extractor.construct_svm_feaset(preptrainingdata, word_features)

        problem = svm_problem(trainingset['labels'], trainingset['feature_vectors'])
        param = svm_parameter('-q')
        param.kernel_type = LINEAR

        start = time.time()
        svm_classifier = svm_train(problem, param)
        svm_trainingtime = time.time() - start

        print "SVM training time:", svm_trainingtime

        svm_save_model('./data/processed/svm_classifier.model', svm_classifier)

    p_labels, p_accs, p_vals = svm_predict(testset['labels'], testset['feature_vectors'], svm_classifier)

    print p_labels
    print p_accs

if __name__ == "__main__":
    main()
