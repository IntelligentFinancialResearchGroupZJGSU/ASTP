import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_packing import shap_loder
import config
import shap
from sklearn.ensemble import GradientBoostingClassifier
from build_edg import build_edge
device = config.device

train_file = config.datafile_source+'/train_data.csv'
test_file = config.datafile_source+'/test.csv'

from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt
import joblib

if __name__ == '__main__':
    n_samples = 50
    n_timesteps = 1019
    n_features = 32
    X_train, y_train = shap_loder(train_file)
    X_train = X_train[:,:-1,:]
    y_train = y_train[:, 1:]
    X_test, _ = shap_loder(test_file)
    x_test = np.array(X_test.reshape(-1, n_features))
    feature_names = ['open', 'high', 'low', 'close', 'volume', 'turnoverRatio', 'dx', 'adx', 'adxr', 'cci', 'wr_5', 'mfi_14', 'mfi_5', 'pdi',
                                      'close_20_sma', 'close_5_sma', 'close_20_ema', 'close_5_ema', 'close_60_ema', 'change', 'cr',
                                      'cr-ma1', 'cr-ma2', 'cr-ma3', 'chop_14', 'chop_5', 'close_5_mstd', 'wt1', 'wt2', 'change_-7_s',
                                      'change_-8_s', 'change_-9_s']
    # feature_names = ['dx', 'adx', 'adxr', 'cci', 'wr_5',
    #                  'mfi_14', 'mfi_5', 'pdi',
    #                  'close_20_sma', 'close_5_sma', 'close_20_ema', 'close_5_ema', 'close_60_ema', 'change', 'cr',
    #                  'cr-ma1', 'cr-ma2', 'cr-ma3', 'chop_14', 'chop_5', 'close_5_mstd', 'wt1', 'wt2', 'change_-7_s',
    #                  'change_-8_s', 'change_-9_s']
    gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    X_trains = X_train.reshape(-1, n_features)
    y_trains = y_train.reshape(-1)
    gbdt.fit(X_trains, y_trains)
    explainer = shap.TreeExplainer(gbdt)
    shap_values = explainer.shap_values(x_test)

    plt.figure(figsize=(10,10))
    shap.summary_plot(shap_values, X_test.reshape((-1,n_features)),feature_names=feature_names,max_display=30,show=False)
    plt.savefig('shap_300.pdf')
    plt.show()
    shap_avg = shap_array.mean(axis=1)
    x_tests = X_test.mean(axis=1)
    print(x_tests.shape)
    shap.summary_plot(shap_avg,x_tests ,feature_names=feature_names)

    //You are a senior data
 analysis expert, skilled at uncovering complex, deep rela
tionships between variables. Conduct a deep analysis of the
 following feature set: Which features have a direct linear or
 nonlinear impact on the target? Which features always work
 in combination with others to produce a greater effect? Are
 there any features that act as mediators, transmitting the effect
 of another feature? Do any features moderate the effect of
 other features on the target? Based on this analysis, propose
 2-3 feature interaction combinations that are likely to have
 strong predictive power//



