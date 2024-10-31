# For necessary data processing and calculations
import numpy as np
import pandas as pd

# For reading and writing files
import json
import pickle
import sys
import os
import glob
import shutil
import io
from io import BytesIO
from pathlib import Path
import base64

# For image processing
from PIL import Image, UnidentifiedImageError
import cv2

# For machine learning
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model
from sklearn import neighbors, metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.colors as mcolors

# For tracking programming progress
from tqdm.notebook import tqdm
from collections import Counter
import random
import time

# For data visualizations
title_font = {'family': 'Palatino',
        'weight': 'normal',
        'size': 16,
        }
axis_font = {'family': 'Palatino',
        'weight': 'light',
        'size': 12,
        }
xticks_font = {'family': 'Helvetica',
        'weight': 'light',
        'color': 'darkgrey',
        'size': 9,
        }
yticks_font = {'family': 'Helvetica',
        'weight': 'normal',
        'color': 'darkgrey',
        'size': 9,
        }
legend_font = {'family': 'Palatino', 
               'size': 10, 
               'weight': 'light',
               'color': 'darkgrey'
              }

green = 'yellowgreen'
blue = 'powderblue'
red = 'lightcoral'

def convert_probabilities(prob_array):
    return [(i, round(prob, 3)) for i, prob in enumerate(prob_array)]

def max_probability_info(prob_array):
    max_index = np.argmax(prob_array)
    max_value = prob_array[max_index]
    return max_index, round(max_value, 3)

def write_classifications_to_folder(input_folder, output_folder, classified_df, threshold):
    '''
    Function to write the classified news images to labelled folders 
    in preparation of manual inspection
    :param input_folder: Input folder in which inclassified images were stored
    :param output_folder: Base output directory to which images must be written
    :param classified_df: Dataframe storing the classifications
    '''
    for index, row in classified_df.iterrows():
        item = row['image_filename']
        label = row[f'pred_label_{threshold}']
        
        # Define origin path, destination dir, destination path
        origin_path = os.path.join(input_folder, item)
        destination_dir = os.path.join(output_folder, label)
        destination_path = os.path.join(destination_dir, item)
        
        # Ensure the destination directory exists
        os.makedirs(destination_dir, exist_ok=True)
        
        # Copy image from input folder to output folder
        shutil.copy(origin_path, destination_path)

def list_corrected_classes(dataset, threshold, data_folder):
    '''
    Function to list the correct labels based on the manually corrected classifications
    :param data_folder: the data folder structure that stores 
    the subfolders which each contain the labelled images
    '''
    classification_labels, article_ids, article_filenames = [], [], []
    for subfolder in os.listdir(data_folder):
        if subfolder.startswith('.') or subfolder.startswith('_'):
            continue
            
        subfolder_path = os.path.join(data_folder, subfolder)
        
        for item in os.listdir(subfolder_path):
            if item.startswith('.'):
                continue
                
            elif item.endswith('.jpg'):
                classification_label = subfolder
                article_id = item.split('_')[0]
                article_filename = item
                
                classification_labels.append(classification_label)
                article_ids.append(article_id)
                article_filenames.append(article_filename)
                
        df = pd.DataFrame({
            'dataset': dataset,
            'id': article_ids,
            'image_filename': article_filenames,
            f'true_label_{threshold}': classification_labels,
        })
    return df

def obtain_metrics_for_thresholds(true_labels, predicted_labels_list):
    '''
    Method that calculates accuracy metrics for dictionaries that store
    parameter values for threshold, label predictions, and the confusion  matrix for those values. 
    :param predicted_labels_list: list with dictionaries that store threshold values, labels and the cm
    '''
    results = []
    for predictions in predicted_labels_list:
        threshold = predictions['threshold']
        predicted_labels = predictions['labels']
        cm = predictions['cm']
        
        accuracy_multiclass = accuracy_score(true_labels, predicted_labels)
        precision_multiclass = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
        recall_multiclass = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
        f1_multiclass = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
        
        false_negatives = cm.sum(axis=1) - np.diag(cm)
        true_positives = np.diag(cm)
        false_negative_rate = false_negatives.sum() / (false_negatives.sum() + true_positives.sum())
    
        results_multiclass = {
                    'Classification': 'Multiclass',
                    'Num_dim': 512,
                    'Threshold': threshold,
                    'Class_Precision': [(i, round(score, 2)) for i, score in enumerate(precision_multiclass)],  
                    'Class_Recall': [(i, round(score, 2)) for i, score in enumerate(recall_multiclass)],     
                    'Class_F1-score': [(i, round(score, 2)) for i, score in enumerate(f1_multiclass)],
            
                    'Accuracy': round(accuracy_multiclass, 2),
                    'Precision': round(precision_multiclass.mean(), 2),
                    'Recall': round(recall_multiclass.mean(), 2),
                    'F1-score': round(f1_multiclass.mean(), 2), 
                    'FN_rate': round(false_negative_rate, 2)
                }
        results.append(results_multiclass)
    return results