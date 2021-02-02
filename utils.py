import pandas as pd
from sklearn.externals.joblib import Memory
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from numpy import savetxt
import warnings

import numpy as np

def dataLoading(path):
    # loading data
    df = pd.read_csv(path)

    labels = df['class']

    x_df = df.drop(['class'], axis=1)

    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)

    return x, labels

def dataLoading_np(path):
    # loading data
   
    x = np.load(path) 

    labels = np.load('../dataset/y_train_1w_50percent.npy')
   
    return x, labels


def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    
    return roc_auc, ap, 

def writeResults(name, losses, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers, depth, rauc, ap, std_auc, std_ap,
                 train_time, test_time, architecture, epochs, batch_size, nb_batch, precision_new, recall_new, f1_new,max_precision, max_recall, f_one, path="../results/robresultsss.csv"):
    csv_file = open(path, 'a')
    row = name + "," + str(losses) + "," + str(n_samples) + "," + str(dim) + ',' + str(n_samples_trn) + ',' + str(
        n_outliers_trn) + ',' + str(n_outliers) + ',' + str(depth) + "," + str(rauc) + "," + str(std_auc) + "," + str(
        ap) + "," + str(std_ap) + "," + str(train_time) + "," + str(test_time)+ "," + str(architecture) + "," +str(batch_size) + "," + str(nb_batch) + "," + str(epochs)+ "," + str(precision_new) + "," + str(recall_new) + "," + str(f1_new) + "," + str(max_precision) + "," + str(max_recall) + "," + str(f_one) + "\n"
    csv_file.write(row)
