from devnet_kdd19 import load_model_weight_predict
from utils import aucPerformance, dataLoading
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import warnings

model_path = '/content/gdrive/My Drive/devnet/DevNet/DevNet-master copy 2/model/devnet_/x_train_1w_50percent_512bs_943ko_1d.h5'
network_depth = 1  
input_shape = [1,298]  

x_test =  np.load('/content/gdrive/My Drive/devnet/DevNet/DevNet-master copy 2/dataset/x_test_1w_50percent.npy')

y_test = np.load('/content/gdrive/My Drive/devnet/DevNet/DevNet-master copy 2/dataset/y_test_1w_50percent.npy')

scores = load_model_weight_predict(model_path,
                                   input_shape=input_shape,
                                   network_depth=network_depth,
                                   x_test=x_test)
                                   
preds = scores
class_one = preds > 0.5
predic_class = np.where(class_one == True,1,0)
precision_new = precision_score(y_test, predic_class)
print('precision',precision_new)
recall_new = recall_score(y_test, predic_class)
print('recall',recall_new)
f1_new = 2 * ((precision_new * recall_new) / (precision_new + recall_new ))
print('f1',f1_new)

AUC_ROC, AUC_PR = aucPerformance(scores, y_test)
