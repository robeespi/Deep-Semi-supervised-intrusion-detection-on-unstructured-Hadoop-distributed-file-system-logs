"""
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
"""
import os
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.config.experimental_run_functions_eagerly(True)
tf.random.set_seed(42)
sess = tf.compat.v1.initialize_all_variables()

from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Embedding
import pandas as pd
import argparse
import numpy as np
import sys
from scipy.sparse import vstack, csc_matrix
from utils import dataLoading, dataLoading_np, aucPerformance, writeResults
from sklearn.model_selection import train_test_split
from numpy import savetxt

import time

MAX_INT = np.iinfo(np.int32).max

def mlp_network_3hl(input_shape):
    '''
    Multilayer perceptron with three hidden layers
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01), name='hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01), name='hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01), name='hl3')(intermediate)
    intermediate = Dense(1, activation='linear', name='score')(intermediate)
    return Model(x_input, intermediate)


def mlp_network_1hl(input_shape):
    '''
    Multilayer perceptron with with one hidden layer
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01), name='hl1')(x_input)
    intermediate = Dense(1, activation='linear', name='score')(intermediate)
    return Model(x_input, intermediate)


def lstm_network(input_shape):
    '''
    LSTM network two hidden layers and one dense layer
    '''
    x_input = Input(shape=input_shape)
    intermediate = LSTM(64, activation='relu', return_sequences=True)(x_input)
    intermediate = LSTM(64, activation='relu', return_sequences=False)(intermediate)
    intermediate = Dense(29, activation='relu')(intermediate)
    intermediate = Dense(1, activation='linear', name='score')(intermediate)
    return Model(x_input, intermediate)


def neural_network(input_shape, network_depth):
    '''
    Build the neural network
    '''
    if network_depth == 4:
        model = mlp_network_3hl(input_shape)
    elif network_depth == 2:
        model = mlp_network_1hl(input_shape)
    elif network_depth == 1:
        model = lstm_network(input_shape)
    else:
        sys.exit("The network is not set properly")
    rms = RMSprop(clipnorm=1.)
    model.compile(loss='binary_crossentropy', optimizer=rms)
    return model


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator technique
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
    counter = 0
    while 1:
        #if data_format == 0:
        ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        #else:
            #ref, training_labels = input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size,rng)
        counter += 1
        yield (ref, training_labels)
        if (counter > nb_batch):
            counter = 0


def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples
    Alternates between positive and negative.
    '''
    dim = x_train.shape[1]
    ke=x_train.shape[2]
    ref = np.empty((batch_size, dim, ke))
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):
        if (i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]

    return np.array(ref), np.array(training_labels)

def load_model_weight_predict(model_name, input_shape, network_depth, x_test):
    '''
    load the saved weights to make predictions
    '''
    model = neural_network(input_shape, network_depth)
    model.load_weights(model_name)
    scoring_network = Model(inputs=model.input, outputs=model.output)
    scores = scoring_network.predict(x_test)
    return scores

def run_t(args):
    
    names = ['x_train_1w_50percent']
    network_depth = int(args.network_depth)
    random_seed = args.ramdn_seed
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)
        filename = nm.strip()
       
        x, labels = dataLoading_np(args.input_path + filename + ".npy")
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]
        n_outliers_org = outliers.shape[0]

        train_time = 0
        test_time = 0
        for i in np.arange(runs):
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42, stratify=labels)
            print('x_train', x_train.shape, type(x_train))
            print('y_train',y_train.shape, type(y_train))
            print('x_test', x_test.shape, type(x_test))
            print('y_test', y_test.shape, type(y_test))
            
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            n_outliers = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            rng = np.random.RandomState(random_seed)
            if n_outliers > args.known_outliers:
                mn = n_outliers - args.known_outliers
                remove_idx = rng.choice(outlier_indices, mn, replace=False)
                x_train = np.delete(x_train, remove_idx, axis=0)
                y_train = np.delete(y_train, remove_idx, axis=0)
        
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            
            print('training samples num:', y_train.shape[0],
                  'outlier num:', outlier_indices.shape[0],
                  'inlier num:', inlier_indices.shape[0])
            input_shape = x_train.shape[1:]
            n_samples_trn = x_train.shape[0]
            n_outliers = len(outlier_indices)
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

            start_time = time.time()
            input_shape = x_train.shape[1:]
            epochs = args.epochs
            batch_size = args.batch_size
            nb_batch = args.nb_batch

            model = neural_network(input_shape, network_depth)
            print(model.summary())
            model_filename= filename + "_" + str(args.batch_size) +"bs_" + str(args.known_outliers) + "ko_" + str(network_depth) +"d.h5" 
            model_name = os.path.join('../model/train_', model_filename)
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                           save_best_only = True, save_weights_only = True)  
            model.fit_generator(
                batch_generator_sup(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng), steps_per_epoch=nb_batch, epochs=epochs, callbacks=[checkpointer])
                
            train_time += time.time() - start_time
            start_time = time.time()
            scores = load_model_weight_predict(model_name, input_shape, network_depth, x_test)
            test_time += time.time() - start_time
            print(scores.shape)
            rauc[i], ap[i] = aucPerformance(scores, y_test)
            preds = scores
            class_one = preds > 0.5
            predic_class = np.where(class_one == True,1,0)
            precision_new = precision_score(y_test, predic_class)
            print('new precision',precision_new)
            recall_new = recall_score(y_test, predic_class)
            print('new recall',recall_new)
            f1_new = 2 * ((precision_new * recall_new) / (precision_new + recall_new ))
            print('f1 new',f1_new)

            fig3 = plt.figure()
            plt.plot(model.history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            fig3.savefig('my_figure3.png')
            
           
            
        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time = train_time / runs
        test_time = test_time / runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
        print("average runtime: %.4f seconds" % (train_time + test_time))
        architecture = '2 hidlstm(64)+1dense(32)+ 1dense(1)'
        losses = 'binary-cross-entropy'
        max_precision=0
        max_recall=0
        f_one=0
        writeResults(filename + '_' + str(network_depth),losses, x.shape[0], x.shape[1], n_samples_trn, n_outliers_org,
                     n_outliers, network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, architecture, epochs, batch_size, nb_batch, precision_new, recall_new, f1_new, max_precision, max_recall, f_one, path=args.output)
                     
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_depth", choices=['1', '2', '4'], default='1',
                        help="the depth of the network architecture")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
    parser.add_argument("--nb_batch", type=int, default=30, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
    parser.add_argument("--runs", type=int, default=1,
                        help="number experiments to obtain the average performance")
    parser.add_argument("--known_outliers", type=int, default=384,
                        help="the number of labeled outliers")
    parser.add_argument("--input_path", type=str, default='../dataset/', help="the path of the data sets")
  
    parser.add_argument("--data_format", choices=['0', '1'], default='0',
                  help="specify whether the input data is a csv (0) or libsvm (1) data format")
    parser.add_argument("--output", type=str,
                        default='../results/robresultsss.csv',
                        help="the output file path")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    args = parser.parse_args()
    run_t(args)
    


    
