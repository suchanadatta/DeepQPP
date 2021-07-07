import numpy as np
import keras, os
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class InteractionData:
    # Interaction data of query qid with K top docs -
    # each row vector is a histogram of interaction data for a document
    def __init__(self, qid, dataPathBase):
        self.qid = qid
        histFile = "{}/{}.bert".format(dataPathBase, self.qid)
        histogram = np.genfromtxt(histFile, delimiter=" ")
        self.matrix = histogram[:, 1:]


class PairCmpDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder, batch_size, dim, qterms, bertVector, splitType):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = dim
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.K = qterms
        self.M = bertVector
        self.on_epoch_end()
        self.splitType = splitType

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
        X = [np.empty((self.batch_size, *self.dim)) for i in range(2)]
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            b_data = InteractionData(b_id, self.dataDir)

            w, h = a_data.matrix.shape
            a_data.matrix = a_data.matrix.reshape(w, h, 1)
            b_data.matrix = b_data.matrix.reshape(w, h, 1)

            X[0][i,] = a_data.matrix
            X[1][i,] = b_data.matrix
            Y[i] = paired_instance.class_label

        if self.splitType == 'train':
            return X, Y
        elif self.splitType == 'test':
            return X


class ConvModel:

    def build_siamese(input_shape):
        input_a = Input(shape=input_shape, dtype='float32')
        input_b = Input(shape=input_shape, dtype='float32')

        matrix_encoder = Sequential(name='sequence_1')
        matrix_encoder.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
        matrix_encoder.add(Dense(500))
        matrix_encoder.add(MaxPooling2D(padding='same'))
        matrix_encoder.add(Flatten())
        matrix_encoder.add(Dropout(0.2))
        matrix_encoder.add(Dense(128, activation='relu'))

        encoded_a = matrix_encoder(input_a)
        encoded_b = matrix_encoder(input_b)

        merged_vector = concatenate([encoded_a, encoded_b], axis=-1, name='concatenate')
        predictions = Dense(1, activation='sigmoid')(merged_vector)

        siamese_net = Model([input_a, input_b], outputs=predictions)
        return siamese_net


class QPPEvalPair:

    def __init__(self, config):
        self.conf = config
        # parameters : K=no. of max qterms
        #              M=bert vector size (default=768)
        #              NumChannel=no. of channels passed through (default=1)
        self.K = 5
        self.M = 768
        self.NUMCHANNELS = 1

        self.BATCH_SIZE = self.conf.config.get('Section', 'batchSize')
        self.EPOCHS = self.conf.config.get('Section', 'epochs')

    def qpp_eval_pair(self, kfold, all_query_pairs_list):
        # K-fold Cross Validation model evaluation
        fold_no = 1
        accuracy_score_list = []
        for train_split, test_split in kfold.split(all_query_pairs_list):
            print('=========== Started new fold ==========')
            print('No. of data points in train split : ', len(train_split))
            print('No. of data points in test split : ', len(test_split), '\n')
            ground_truth = []
            predict_list = []

            # build siamese model
            siamese_model = ConvModel.build_siamese((self.K, self.M, self.NUMCHANNELS))
            siamese_model.compile(loss=keras.losses.BinaryCrossentropy(),
                                  optimizer=keras.optimizers.Adam(),
                                  metrics=['accuracy'])
            siamese_model.summary()

            # train data generator
            training_generator = PairCmpDataGenerator(np.array(all_query_pairs_list)[train_split],
                                                            dataFolder=self.conf.config.get('Section', 'bertVecPath'),
                                                            batch_size=int(self.BATCH_SIZE),
                                                            dim=(self.K, self.M, self.NUMCHANNELS),
                                                            qterms=self.K, bertVector=self.M,
                                                            splitType='train')
            print('Size of the training generator : ', len(training_generator))

            # learn model parameters with the train split
            siamese_model.fit_generator(generator=training_generator,
                                        use_multiprocessing=True,
                                        epochs=int(self.EPOCHS),
                                        workers=4)
                                        # validation_split=0.2,
                                        # verbose=1)

            # save model weights (uncomment this if you wish to save each fold's weights)
            # model_dir = 'pair_model_weights/'
            # if not os.path.exists(model_dir):
            #     os.makedirs(model_dir)
            # siamese_model_custom.save_weights(model_dir + 'fold-' + str(fold_no) + '.weights')

            # generate the ground truth for current test split
            for pair_instance in np.array(all_query_pairs_list)[test_split]:
                ground_truth.append(pair_instance.class_label)
            # print('GT : ', ground_truth)

            # test data generator
            test_generator = PairCmpDataGenerator(np.array(all_query_pairs_list)[test_split],
                                                       dataFolder=self.conf.config.get('Section', 'interMatrixPath'),
                                                       batch_size=int(self.BATCH_SIZE),
                                                       dim=(self.K, self.M, self.NUMCHANNELS),
                                                       qterms=self.K, bertVector=self.M,
                                                       splitType='test')
            print('Size of the test generator : ', len(test_generator))

            # make predictions
            predictions = siamese_model.predict(test_generator)
            print('Shape of the predicted matrix : ', predictions.shape)

            # write predicted values to a file
            if self.conf.config.get('Section', 'result') == 'yes':
                QPPEvalPair.write_predict_label(fold_no, predictions, test_generator)

            i = 0
            for entry in test_generator.paired_instances_ids:
                if predictions[i] >= 0.45:
                    # print(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i], 4)) + '\t' + '1')
                    predict_list.append(round(1))
                else:
                    # print(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i], 4)) + '\t' + '0')
                    predict_list.append(0)
                i += 1

            score = accuracy_score(ground_truth, predict_list)
            print('Accuracy gained with this fold : ', round(score, 4))
            accuracy_score_list.append(score)
            fold_no += 1

        print('Accuracy obtained from different fold : ', accuracy_score_list)
        print('5-fold CV accuracy : ', float(sum(accuracy_score_list)) / float(len(accuracy_score_list)))

    @staticmethod
    def write_predict_label(fold_no, predictions, test_generator):
        cwd = os.getcwd() + '/pair_predicted_res/'
        if not os.path.exists(cwd):
            os.makedirs(cwd)

        with open(cwd + 'fold-' + str(fold_no) + '.predict', 'w') as outFile:
            i = 0
            for entry in test_generator.paired_instances_ids:
                if predictions[i][0] >= 0.45:
                    outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '1\n')
                else:
                    outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '0\n')
                i += 1
        outFile.close()