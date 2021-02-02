from train import load_model_weight_predict
from utils import aucPerformance, dataLoading, prec
import numpy as np
import warnings

model_path = '../model/train_/x_train_1w_50percent_512bs_384ko_1d.h5'
network_depth = 1  
input_shape = [1,298]  

x_test =  np.load('../dataset/x_test_1w_50percent.npy')

y_test = np.load('../dataset/y_test_1w_50percent.npy')

scores = load_model_weight_predict(model_path,
                                   input_shape=input_shape,
                                   network_depth=network_depth,
                                   x_test=x_test)

AUC_ROC, AUC_PR = aucPerformance(scores, y_test)

max_prec, max_rec, f1 = prec(scores, y_test)
