from FlightMLP import FlightMLP
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from FlightDataset import FlightDataset


def model_eval(y_true, y_pred):
    '''
    y_true: true labels
    y_pred: sigmoid output
    '''
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print('precision: ', precision[1])
    print('recall: ', recall[1])
    print('f1: ', f1[1])
    print('accuracy', accuracy)

    
    