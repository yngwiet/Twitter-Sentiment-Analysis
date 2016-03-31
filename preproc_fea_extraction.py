#!/usr/bin/python

import csv
import itertools
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


class Preprocessor:
    def __init__(self, origintrainloc, origintestloc):
        self.origintrainloc = origintrainloc
        self.origintestloc = origintestloc

    def read_data(self, numofpos, numofneg):
        f = open(self.origintrainloc, 'r')
        f_csv = csv.reader(f)

        trainingdata = []

        count = 0

        for row in f_csv:
            count += 1
            if count <= 800000:
                trainingdata.append([row[5], 'neg'])
            else:
                trainingdata.append([row[5], 'pos'])

        f.close()

        # get tweets and their sentiment labels from test dataset
        f = open(self.origintestloc, 'r')
        f_csv = csv.reader(f)

        testdata = []

        for row in f_csv:
            if row[0] != '2':  # ignore neutral test data
                if row[0] == '0':
                    testdata.append([row[5], 'neg'])
                else:
                    testdata.append([row[5], 'pos'])

        f.close()

        # set desired nunmber of training data here !!!
        sampletraining = trainingdata[:numofneg] + trainingdata[1600000-numofpos:]

        return sampletraining, testdata

    @staticmethod
    def preprocess(document):
        document = unicode(document, errors='ignore')
        document = document.lower()
        document = re.sub(r'http\S*', '', document)  # urls
        document = re.sub(r'@\S*', '', document)  # @ tag
        document = re.sub(r'#\S*', '', document)  # hash tags
        document = re.sub(r'(\w)\1{2,}', r'\1\1', document)  # e.g. looovvveee -> loovvee
        document = re.sub(r'\.{2}', ' ', document)  # e.g. 'word..' or '..word'
        document = re.sub(r'[/\*~\|\^\\]', ' ', document)  # e.g. 'my/our' and '*real star*' and 'ha~', etc

        return document


class FeatureExtractor:
    nltk_stop_words = set(stopwords.words('english'))  # default stop words list
    porterstemmer = PorterStemmer()  # default stemmer
    wordnetlemmatizer = WordNetLemmatizer()  # default lemmatizer

    def __init__(self, stop_words=None, stemmer=None, lemmatizer=None):
        if stop_words is None:
            self.stop_words = FeatureExtractor.nltk_stop_words
        else:
            self.stop_words = stop_words

        if stemmer is None:
            self.stemmer = FeatureExtractor.porterstemmer
        else:
            self.stemmer = stemmer

        if lemmatizer is None:
            self.lemmatizer = FeatureExtractor.wordnetlemmatizer
        else:
            self.lemmatizer = lemmatizer

    def get_feavector(self, document):

        words = word_tokenize(document)

        # remove special chars. e.g. punctuation
        words = [word for word in words
                 if re.match(r'^["\'&/,\.\?!:;\|\$%\^\*\+=`~\\\(\)\[\]\{\}<>_\-]+$', word) is None]

        # get rid of words that contain numbers
        words = [word for word in words
                 if re.match(r'.*\d+.*', word) is None]

        # remove stop words and html entity word
        words = [word for word in words if word not in self.stop_words and word != 'quot']

        # apply lemmatizing
        words = [self.lemmatizer.lemmatize(word) for word in words]

        # apply stemming
        words = [self.stemmer.stem(word) for word in words]

        return words

    def get_bigram_feavector(self, document):
        words = self.get_feavector(document)
        return list(bigrams(words))

    @staticmethod
    def get_features(all_words, feanum):
        freqdist = nltk.FreqDist(all_words)

        if feanum == 'max':
            feanum = len(list(freqdist.keys()))
        elif feanum > len(list(freqdist.keys())):
            feanum = len(list(freqdist.keys()))

        print "the number of features is", feanum

        featuples = freqdist.most_common(feanum)

        features = []

        for i in range(feanum):
            features.append(featuples[i][0])

        return features

    @staticmethod
    def get_unibigram_features(all_words, uni_feanum, bi_feanum):
        word_fd = nltk.FreqDist(all_words)
        bigram_fd = nltk.FreqDist(nltk.bigrams(all_words))

        if uni_feanum == 'max':
            uni_feanum = len(list(word_fd.keys()))
        elif uni_feanum > len(list(word_fd.keys())):
            uni_feanum = len(list(word_fd.keys()))

        if bi_feanum == 'max':
            bi_feanum = len(list(bigram_fd.keys()))
        elif bi_feanum > len(list(bigram_fd.keys())):
            bi_feanum = len(list(bigram_fd.keys()))

        finder = BigramCollocationFinder(word_fd, bigram_fd)
        bigrams = finder.nbest(BigramAssocMeasures.chi_sq, bi_feanum)

        print "the number of unigram features is", uni_feanum
        print "the number of bigram features is", bi_feanum

        featuples = word_fd.most_common(uni_feanum)

        selected_words = []

        for i in range(uni_feanum):
            selected_words.append(featuples[i][0])

        features = []
        for ngram in itertools.chain(selected_words, bigrams):
            features.append(ngram)

        return features

    def construct_feaset(self, document, word_features):
        features = {}
        words = self.get_feavector(document)
        for word in word_features:
            features[word] = (word in words)

        return features

    def construct_unibigram_feaset(self, document, word_features):
        features = {}
        words = self.get_feavector(document)
        bigram_words = self.get_bigram_feavector(document)
        for word in word_features:
            features[word] = (word in words+bigram_words)

        return features

    def construct_svm_feaset(self, prepdata, word_features):
        word_features = sorted(word_features)
        feature_vectors = []
        labels = []
        for row in prepdata:
            label = 0
            map = {}
            for word in word_features:
                map[word] = 0
            words = self.get_feavector(row[0])
            for word in words:
                if word in map:
                    map[word] = 1
            values = map.values()
            feature_vectors.append(values)
            if row[1] == 'pos':
                label = 0
            elif row[1] == 'neg':
                label = 1
            labels.append(label)
        return {'feature_vectors': feature_vectors, 'labels':labels}

    def construct_unibigram_svm_feaset(self, prepdata, word_features):
        word_features = sorted(word_features)
        feature_vectors = []
        labels = []
        for row in prepdata:
            label = 0
            map = {}
            for word in word_features:
                map[word] = 0
            words = self.get_feavector(row[0])
            bigram_words = self.get_bigram_feavector(row[0])
            for word in words+bigram_words:
                if word in map:
                    map[word] = 1
            values = map.values()
            feature_vectors.append(values)
            if row[1] == 'pos':
                label = 0
            elif row[1] == 'neg':
                label = 1
            labels.append(label)
        return {'feature_vectors': feature_vectors, 'labels': labels}
