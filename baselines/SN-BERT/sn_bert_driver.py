import sys, os, random, configparser
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import pair_eval as pair
import point_eval as point

import re
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
from transformers import pipeline
from transformers import RobertaTokenizer, TFRobertaModel


seed_value = 12321
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

class Config:

    def __init__(self, prop_file):
        self.prop_file = prop_file
        self.config = configparser.ConfigParser()
        self.config.read(self.prop_file)


class QueryBERT:

    def __init__(self, prop_file):
        self.conf = Config(prop_file)
        self.query_dict = {}
        self.queryFile = self.conf.config.get('Section', 'queryPath')

    def make_qterm_dict(self):
        stemmer = PorterStemmer()
        max_qterm = 0
        rootElement = ET.parse(self.queryFile).getroot()
        for subElement in rootElement:
            query = re.sub('[^a-zA-Z0-9\n\.]', ' ', subElement[1].text)
            query_terms = query.split()
            if len(query_terms) > max_qterm:
                max_qterm = len(query_terms)
            qterm_list = []
            for term in query_terms:
                qterm_list.append(stemmer.stem(term.lower().strip()))
            self.query_dict[subElement[0].text.strip()] = qterm_list
        return max_qterm

    def vec2str(self, x):
        vecstr = ''
        for x_i in x:
            x_i_str = '%.4f' % (x_i)
            vecstr += x_i_str + ' '
        return vecstr[0:-1]  # strip off the trailing space

    def make_query_vectors(self):
        max_qterm = self.make_qterm_dict()
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = TFRobertaModel.from_pretrained('roberta-base')
        nlp_features = pipeline('feature-extraction')
        for qid in self.query_dict:
            qterms = self.query_dict.get(qid)
            if len(qterms) == max_qterm:
                with open(self.conf.config.get('Section', 'bertVecPath') + qid + '.bert', 'w') as f:
                    for t in qterms:
                        output = nlp_features(t)
                        output = np.array(output)
                        output = output[0]
                        vec = output[1]  # middle vector, first is bos, last is eos
                        vecstr = self.vec2str(vec)
                        f.write(t + ' ' + vecstr + '\n')
                f.close()
                print('Write BERT vectors for qid : ', qid)
            else:
                diff = max_qterm - len(qterms)
                with open(self.conf.config.get('Section', 'bertVecPath') + qid + '.bert', 'w') as f:
                    for t in qterms:
                        output = nlp_features(t)
                        output = np.array(output)
                        output = output[0]
                        vec = output[1]  # middle vector, first is bos, last is eos
                        vecstr = self.vec2str(vec)
                        f.write(t + ' ' + vecstr + '\n')
                    while diff > 0:
                        f.write('x' + ' ')
                        for i in range(1, 769):
                            f.write('0.0' + ' ')
                        f.write('\n')
                        diff = diff - 1
                f.close()
                print('Write BERT vectors for qid : ', qid)
        return max_qterm


class PairedInstance:

    def __init__(self, line):
        l = line.strip().split('\t')
        if len(l) > 2:
            self.qid_a = l[0]
            self.qid_b = l[1]
            self.class_label = int(l[2])
        else:
            self.qid_a = l[0]
            self.qid_b = l[1]

    def __str__(self):
        return '({}, {})'.format(self.qid_a, self.qid_b)

    def getKey(self):
        return '{}-{}'.format(self.qid_a, self.qid_b)


# Separate instances for training/test sets etc. Load only the id pairs.
# Data is loaded later in batches with a subclass of Keras generator
class PairedInstanceIds:
    '''
    Each line in this file should comprise three tab separated fields
    <id1> <id2> <label (1/0)>
    '''

    def __init__(self, idpairLabelsFile):
        self.data = {}
        with open(idpairLabelsFile) as f:
            content = f.readlines()

        # remove whitespace characters like `\n` at the end of each line
        for x in content:
            instance = PairedInstance(x)
            self.data[instance.getKey()] = instance

class SNBERT:

    def __init__(self, prop_file):
        self.conf = Config(prop_file)
        self.k_fold = self.conf.config.get('Section', 'cvFolds')

    def make_query_pairs(self):
        all_query_pairs = PairedInstanceIds(self.conf.config.get('Section', 'qPairJudge'))
        all_query_pairs_list = list(all_query_pairs.data.values())
        print('{}/{} pairs for training'.format(len(all_query_pairs_list), len(all_query_pairs_list)))
        return all_query_pairs_list

    def conv_qpp_eval(self):
        # create query-query pairs
        all_query_pairs_list = SNBERT.make_query_pairs(self)

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=int(self.k_fold), shuffle=True)

        if self.conf.config.get('Section', 'evalType') == 'pair':
            print('\n========= Pairwise Evaluation =========')
            pair.QPPEvalPair.qpp_eval_pair(pair.QPPEvalPair(Config(sys.argv[1])), kfold, all_query_pairs_list)
        elif self.conf.config.get('Section', 'evalType') == 'point':
            print('======== Pointwise Evaluation =========')
            point.QPPEvalPoint.qpp_eval_point(point.QPPEvalPoint(Config(sys.argv[1])), kfold, all_query_pairs_list)
        else:
            print('Choose an evaluation option - pair / point')
            exit(0)

# qbert = QueryBERT(sys.argv[1])
# qbert.make_query_vectors()
SNBERT(sys.argv[1]).conv_qpp_eval()
