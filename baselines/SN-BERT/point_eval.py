import numpy as np
import keras, os
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Lambda
from keras.layers.merge import concatenate
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from keras.optimizers import Adam
import pair_eval as pair
import eval_rank_corr as eval


class PointInstance:
    def __init__(self, id):
        self.qid_a = id

    def __str__(self):
        return "({})".format(self.qid_a)

    def getKey(self):
        return "{}".format(self.qid_a)


class PointInstanceIds:
    def __init__(self, testSet):
        self.data = {}
        for id in testSet:
            instance = PointInstance(id)
            self.data[instance.getKey()] = instance


class PairCmpDataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder, batch_size, dim_top, dim_bottom, dim_label, topDocs,
                 bottomDocs, interMatrix):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = [dim_top, dim_bottom, dim_top, dim_bottom, dim_label]
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.K = topDocs
        self.L = bottomDocs
        self.M = interMatrix
        self.on_epoch_end()
        self.totalRetDocs = 100

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / int(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim[i])) for i in range(5)]
        Y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = pair.InteractionData(a_id, self.dataDir)
            a_data_top = a_data.matrix[0:self.K, :]
            a_data_bottom = a_data.matrix[(self.totalRetDocs - self.L):, :]
            # assert a_data_top.shape != (self.K, self.M), a_data_top.shape

            b_data = pair.InteractionData(b_id, self.dataDir)
            b_data_top = b_data.matrix[0:self.K, :]
            b_data_bottom = b_data.matrix[(self.totalRetDocs - self.L):, :]
            # assert b_data_bottom.shape != (self.K, self.M), b_id

            # w, h = a_data.matrix.shape
            w_top_a, h_top_a = a_data_top.shape
            w_bottom_a, h_bottom_a = a_data_bottom.shape
            a_data_top = a_data_top.reshape(w_top_a, h_top_a, 1)
            a_data_bottom = a_data_bottom.reshape(w_bottom_a, h_bottom_a, 1)

            w_top_b, h_top_b = b_data_top.shape
            w_bottom_b, h_bottom_b = b_data_bottom.shape
            b_data_top = b_data_top.reshape(w_top_b, h_top_b, 1)
            b_data_bottom = b_data_bottom.reshape(w_bottom_b, h_bottom_b, 1)

            X[0][i,] = a_data_top
            X[1][i,] = a_data_bottom
            X[2][i,] = b_data_top
            X[3][i,] = b_data_bottom
            X[4][i,] = paired_instance.class_label
            Y[i] = paired_instance.class_label

        return X, Y


class PointCmpDataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder, batch_size, dim_top, dim_bottom, topDocs,
                 bottomDocs, interMatrix):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = [dim_top, dim_bottom]
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.K = topDocs
        self.L = bottomDocs
        self.M = interMatrix
        self.on_epoch_end()
        self.totalRetDocs = 100

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim[i])) for i in range(1)]

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a

            # read from the data file and construct the instances
            a_data = pair.InteractionData(a_id, self.dataDir)
            a_data_top = a_data.matrix[0:self.K, :]
            a_data_bottom = a_data.matrix[(self.totalRetDocs - self.L):, :]

            w_top_a, h_top_a = a_data_top.shape
            w_bottom_a, h_bottom_a = a_data_bottom.shape
            a_data_top = a_data_top.reshape(w_top_a, h_top_a, 1)
            a_data_bottom = a_data_bottom.reshape(w_bottom_a, h_bottom_a, 1)

            X[0][i,] = a_data_top
            # X[1][i,] = a_data_bottom

        return X


class ConvModel:

    def pair_loss(x):
        # Pair Loss function.
        query1, query2, label = x
        hinge_margin = 1
        keras.backend.print_tensor(query1)
        max_margin_hinge = hinge_margin - label * (query1 - query2)
        loss = keras.backend.maximum(0.0, max_margin_hinge)
        return loss

    def identity_loss(y_true, y_pred):
        return keras.backend.mean(y_pred)

    def base_model(input_shape):
        matrix_encoder = Sequential(name='sequence')
        matrix_encoder.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
        # matrix_encoder.add(Dense(500))
        matrix_encoder.add(MaxPooling2D(padding='same'))
        matrix_encoder.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        # matrix_encoder.add(Dense(500))
        matrix_encoder.add(MaxPooling2D(padding='same'))
        matrix_encoder.add(Flatten())
        matrix_encoder.add(Dropout(0.2))
        matrix_encoder.add(Dense(128, activation='relu'))
        matrix_encoder.add(Dense(1, activation='sigmoid'))

        return matrix_encoder

    def build_siamese_custom_loss(input_shape_top, input_shape_bottom, input_label_shape, base_model):
        input_a_top = Input(shape=input_shape_top, dtype='float32')
        input_a_bottom = Input(shape=input_shape_bottom, dtype='float32')
        # input_c_top = Input(shape=input_label_shape, dtype='float32')

        input_b_top = Input(shape=input_shape_top, dtype='float32')
        input_b_bottom = Input(shape=input_shape_bottom, dtype='float32')
        input_c = Input(shape=input_label_shape, dtype='float32')

        matrix_encoder_top = Sequential(name='sequence_1')
        matrix_encoder_top.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape_top))
        # matrix_encoder_top.add(Dense(500))
        matrix_encoder_top.add(MaxPooling2D(padding='same'))
        matrix_encoder_top.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape_top))
        # matrix_encoder_top.add(Dense(500))
        matrix_encoder_top.add(MaxPooling2D(padding='same'))
        matrix_encoder_top.add(Flatten())
        matrix_encoder_top.add(Dropout(0.2))
        matrix_encoder_top.add(Dense(128, activation='relu'))
        matrix_encoder_top.add(Dense(1, activation='linear'))

        matrix_encoder_bottom = Sequential(name='sequence_2')
        matrix_encoder_bottom.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape_bottom))
        # matrix_encoder_bottom.add(Dense(500))
        matrix_encoder_bottom.add(MaxPooling2D(padding='same'))
        matrix_encoder_bottom.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape_bottom))
        # matrix_encoder_bottom.add(Dense(500))
        matrix_encoder_bottom.add(MaxPooling2D(padding='same'))
        matrix_encoder_bottom.add(Flatten())
        matrix_encoder_bottom.add(Dropout(0.2))
        matrix_encoder_bottom.add(Dense(128, activation='relu'))
        matrix_encoder_bottom.add(Dense(1, activation='linear'))

        encoded_a_top = base_model(input_a_top)
        encoded_a_bottom = base_model(input_a_bottom)
        merged_vector_a = concatenate([encoded_a_top, encoded_a_bottom], axis=-1, name='concatenate_1')

        encoded_b_top = base_model(input_b_top)
        encoded_b_bottom = base_model(input_b_bottom)
        merged_vector_b = concatenate([encoded_b_top, encoded_b_bottom], axis=-1, name='concatenate_2')

        pair_indicator = Lambda(ConvModel.pair_loss)([merged_vector_a, merged_vector_b, input_c])
        siamese_net_custom = Model(inputs=[input_a_top, input_a_bottom, input_b_top, input_b_bottom, input_c],
                                   outputs=pair_indicator)

        return siamese_net_custom


class QPPEvalPoint:

    def __init__(self, config):
        self.conf = config
        # parameters : K=no. of top docs (default=10)
        #              L=no. of bottom docs (default=10)
        #              M=bin size (default=30)
        #              NumChannel=no. of channels passed through (default=1)
        self.K = 10
        self.L = 10
        self.M = 120
        self.NUMCHANNELS = 1
        self.LR = 0.0001
        self.apPath = self.conf.config.get('Section', 'apPath')
        self.BATCH_SIZE = self.conf.config.get('Section', 'batchSize')
        self.EPOCHS = self.conf.config.get('Section', 'epochs')

    def make_ap_dict(self):
        query_ap_dict = {}
        with open(self.apPath) as f:
            content = f.readlines()
            for line in content:
                query_ap_dict[line.strip().split('\t')[0]] = line.strip().split('\t')[1]
        return query_ap_dict

    def create_test_split_GT(query_qp_dict, all_pairs, test_split):
        # create unique query list from the test split
        test_data_set = np.array(all_pairs)[test_split]
        uniq_test_set = set()
        for qid in test_data_set:
            uniq_test_set.add(qid.qid_a)

        # crete the GT for this test split
        test_set_gt = []
        for id in uniq_test_set:
            test_set_gt.append(float(query_qp_dict.get(id)))

        return uniq_test_set, test_set_gt

    def qpp_eval_point(self, kfold, all_query_pairs_list):
        # create query-AP dictionary
        query_ap_dict = QPPEvalPoint.make_ap_dict(self)

        # K-fold Cross Validation model evaluation
        fold_no = 1
        pearsons = 0
        spearmans = 0
        kendalls = 0

        for train_split, test_split in kfold.split(all_query_pairs_list):
            print('=========== Started new fold ==========')
            print('No. of data points in train split : ', len(train_split))
            print('No. of data points in test split : ', len(test_split), '\n')

            # create unique query list from the test split and GT for the same
            uniq_test_set, test_ground_truth = QPPEvalPoint.create_test_split_GT(query_ap_dict, all_query_pairs_list,
                                                                                 test_split)

            # create test generator(compatible to the model)
            all_point_test = PointInstanceIds(uniq_test_set)  # (4)
            all_point_list_test = list(all_point_test.data.values())

            # build siamese model
            base = ConvModel.base_model((self.K, self.M, self.NUMCHANNELS))
            siamese_model_custom = ConvModel.build_siamese_custom_loss((self.K, self.M, self.NUMCHANNELS),
                                                                       (self.L, self.M, self.NUMCHANNELS),
                                                                       (1, 1, 1), base)
            siamese_model_custom.compile(loss=ConvModel.identity_loss,
                                         optimizer=Adam(self.LR),
                                         metrics=['accuracy'])

            siamese_model_custom.summary()

            # train data generator
            training_generator = PairCmpDataGeneratorTrain(np.array(all_query_pairs_list)[train_split],
                                                            dataFolder=self.conf.config.get('Section', 'interMatrixPath'),
                                                            batch_size=int(self.BATCH_SIZE),
                                                            dim_top=(self.K, self.M, self.NUMCHANNELS),
                                                            dim_bottom=(self.L, self.M, self.NUMCHANNELS),
                                                            dim_label=(1,1,1),
                                                            topDocs=self.K, bottomDocs=self.L, interMatrix=self.M)
            print('Size of the training generator : ', len(training_generator))

            # learn model parameters with the train split
            siamese_model_custom.fit_generator(generator=training_generator,
                                        use_multiprocessing=True,
                                        epochs=int(self.EPOCHS),
                                        workers=4)
                                        # validation_split=0.2,
                                        # verbose=1)

            # save model weights (uncomment this if you wish to save each fold's weights)
            # model_dir = 'point_model_weights/'
            # if not os.path.exists(model_dir):
            #     os.makedirs(model_dir)
            # siamese_model_custom.save_weights(model_dir + 'fold-' + str(fold_no) + '.weights')

            # test data generator
            test_generator = PointCmpDataGeneratorTest(all_point_list_test,
                                                       dataFolder=self.conf.config.get('Section', 'interMatrixPath'),
                                                       batch_size=int(self.BATCH_SIZE),
                                                       dim_top=(self.K, self.M, self.NUMCHANNELS),
                                                       dim_bottom=(self.L, self.M, self.NUMCHANNELS),
                                                       topDocs=self.K, bottomDocs=self.L, interMatrix=self.M)
            print('Size of the test generator : ', len(test_generator))

            # make predictions
            predictions = base.predict(test_generator)
            print('Shape of the predicted matrix : ', predictions.shape)
            predict_list = []
            i = 0
            while i < len(predictions):
                predict_list.append(float(predictions[i]))
                i += 1

            # write predicted values to a file
            if self.conf.config.get('Section', 'result') == 'yes':
                QPPEvalPoint.write_predict_label(fold_no, predictions, test_generator)

            # measure rank-correlations
            r, rho, tau = eval.reportRankCorr(test_ground_truth, predict_list)
            pearsons += r
            spearmans += rho
            kendalls += tau
            fold_no +=1

        print('\n======= 5-fold CV rank correlation measures =========')
        print('P-r = {:,.4f}, S-rho = {:.4f}, K-tau = {:.4f}'
              .format(pearsons/(fold_no-1), spearmans/(fold_no-1), kendalls/(fold_no-1)))

    @staticmethod
    def write_predict_label(fold_no, predictions, test_generator):
        cwd = 'point_predicted_res/'
        if not os.path.exists(cwd):
            os.makedirs(cwd)

        with open(cwd + 'fold-' + str(fold_no) + '.predict', 'w') as outFile:
            i = 0
            for entry in test_generator.paired_instances_ids:
                outFile.write(entry.qid_a + '\t' + str(round(predictions[i][0], 4)) + '\n')
                i += 1
        outFile.close()