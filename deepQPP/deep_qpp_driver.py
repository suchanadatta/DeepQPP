import sys, os, random, configparser
import pair_eval as pair
import point_eval as point
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


seed_value = 12321
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

class Config:

    def __init__(self, prop_file):
        self.prop_file = prop_file
        self.config = configparser.ConfigParser()
        self.config.read(self.prop_file)

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

class DeepQPP:

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
        all_query_pairs_list = DeepQPP.make_query_pairs(self)

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

DeepQPP(sys.argv[1]).conv_qpp_eval()