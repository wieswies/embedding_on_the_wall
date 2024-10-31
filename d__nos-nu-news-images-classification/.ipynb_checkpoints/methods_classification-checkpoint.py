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

def evaluate_learners(classifiers, X, y):
    ''' 
    Evaluate each classifier in 'classifiers' with cross-validation on the provided (X, y) data. 
    
    Given a list of classifiers [Classifier1, Classifier2, ..., ClassifierN] return two lists:
     - a list with the scores obtained on the training samples for each classifier,
     - a list with the test scores obtained on the test samples for each classifier.
     The order of scores should match the order in which the classifiers were originally provided. E.g.:     
     [Classifier1 train scores, ..., ClassifierN train scores], [Classifier1 test scores, ..., ClassifierN test scores]
    '''
    # Evaluate with 3-fold cross-validation.
    xvals = [cross_validate(clf, X, y, return_train_score= True, n_jobs=-1) for clf in classifiers]
    train_scores = [x['train_score'] for x in xvals]
    test_scores = [x['test_score'] for x in xvals]
    return train_scores, test_scores

def plot_tuning(grid_search, param_name, ax):
    '''
    Generic plot for 1D grid search
    :grid_search: the result of the GridSearchCV
    :param_name: the name of the parameter that is being varied
    '''
    ax.plot(grid_search.param_grid[param_name], grid_search.cv_results_['mean_test_score'], marker = '.', label = 'Test score')
    ax.plot(grid_search.param_grid[param_name], grid_search.cv_results_['mean_train_score'], marker = '.', label = 'Train score')
    ax.set_ylabel('score (ACC)')
    ax.set_xlabel(param_name)
    ax.legend()
    ax.set_xscale('log')
    #ax.set_title(grid_search.best_estimator_.__class__.__name__, fontdict=title_font)
    if grid_search.best_estimator_.__class__.__name__ == 'SVC':
        ax.set_title(f'{grid_search.best_estimator_.__class__.__name__} ({grid_search.best_estimator_.kernel})', fontdict=title_font)
    else: ax.set_title(f'{grid_search.best_estimator_.__class__.__name__}', fontdict=title_font)
    bp, bs = grid_search.best_params_[param_name], grid_search.best_score_
    ax.text(bp,bs,"  C:{:.2E}, ACC:{:.4f}".format(bp,bs))

def convert_probabilities(prob_array):
    return [(i, round(prob, 3)) for i, prob in enumerate(prob_array)]

def max_probability_info(prob_array):
    max_index = np.argmax(prob_array)
    max_value = prob_array[max_index]
    return max_index, round(max_value, 3)

# Function to save plot to a Base64-encoded string
def save_plot_to_html(fig):
    '''
    To avoid in-line printing, the report is written to HTML
    '''
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%;height:auto;">'

# Plot unknown / politicians probability distribution
def plot_known_unknown_probability_dist(known_probs, unknown_probs, num_dim):
    '''
    Function that plots the distribution of the maximum probabilities 
    for the different classes (here: politicians (knowns) and unknown people (unknowns)).
    Calls save_plot_to_html(fig) to avoid inline printing.
    :param known_probs: Probabilites for items where their class label !=16
    :param unknown_probs: Probabilites for items where their class laben == 16 (i.e. the unknown class)
    '''
    fig, ax = plt.subplots()
    ax.hist(known_probs, bins=50, alpha=0.5, label='Politicians', color='skyblue')
    ax.hist(unknown_probs, bins=50, alpha=0.5, label='Unknowns', color='orange')
    ax.set_xlabel('Maximum Probability')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper center')
    ax.set_title(f'Distribution of Maximum Probabilities in {num_dim} dimensions', fontdict=title_font)
    return save_plot_to_html(fig)

# Plot confusion matrix
def print_confusion_matrices(cm, cm_normalized_df, cm_df, threshold):
    '''
    Function that plots the Confusion Matrix and the normalized Confusion Matrix.
    Calls save_plot_to_html(fig) to avoid inline printing.
    :param cm: Confusion Matrix (sklearn)
    :param cm_normalized_df: Confusion Matrix dataframe, normalized
    :param cm_df: Confusion Matrix dataframe
    :threshold: Threshold for which the matrix is shown
    '''
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Normalized
    sns.heatmap(cm_normalized_df, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Normalized CM ({threshold["t_name"]}: {threshold["t_value"]})', fontdict=title_font)
    # Non-normalized
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar_kws={'label': 'Count'}, vmax=np.percentile(cm, 95))
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'CM ({threshold["t_name"]}: {threshold["t_value"]}) - colorscale adjusted', fontdict=title_font)
    plt.tight_layout()
    return save_plot_to_html(fig)

# Function to train in different dimensions, and evaluate different thresholds for those dimensions
def train_dev_dimensions(pipeline, X_data, y_data, thresholds):
    '''
    Function that explores the model pipeline's performance for
    different embedding data dimensionalities and different cut-off thresholds
    that must identify and classify unknowns from politicians
    (where the unknown class is not in the training data).
    :param pipeline: Model pipeline (here: SVC or kNN)
    :param X_data: Data dictionary storing the different embeddings for different dimensionalities
    :param y_data: Data dictionary storing the labels for training, development and testing
    :param thresholds: Data dictionary storing the thresholds and their values
    '''
    results = []
    html_report = f'<html><head><title style="color:black;font-family:Palatino;text-align:center;"> Model Evaluation Report for {pipeline.steps[1][0]}</title></head><body>'
    
    for dimension in X_data:
        predicted_labels_list = []
        b_predicted_labels_list = []
        html_report += f'<h2 style="color:black;font-family:Palatino;text-align:center;">Training and Development for {dimension["num_dim"]} dimensions</h2>'
        
        # Prepare data infrastructure
        X_train = dimension['X_data']['X_train']
        X_dev = dimension['X_data']['X_dev']
        
        y_train = y_data[0]['y_train']
        y_dev = y_data[0]['y_dev']

        # Fit the training data to the pipeline
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_dev)
        max_prob_info = [max_probability_info(row) for row in probabilities]
        pred_class, max_prob = zip(*max_prob_info)

        # Store the data in a dataframe for easy access
        df_dev = pd.DataFrame({
            'X': X_dev.tolist(),
            'y': y_dev.tolist(),
            'max_prob': max_prob,
            'pred': pred_class
        })
        df_dev['pred'] = df_dev['pred'].apply(lambda x: 17 if x == 16 else x) # correction for missing Unknown class in the training data

        # Inspect probabilities
        known_probs = df_dev[df_dev['y'] != 16]['max_prob']
        unknown_probs = df_dev[df_dev['y'] == 16]['max_prob']
        html_report += plot_known_unknown_probability_dist(known_probs, unknown_probs, dimension['num_dim'])

        # Recalculate classes for different thresholds
        for t in thresholds:
            df_dev[f'pred_{t["t_name"]}'] = df_dev.apply(lambda row: 16 if row['max_prob'] < t['t_value'] else row['pred'], axis=1)
            predicted_labels_list.append(df_dev[f'pred_{t["t_name"]}'])
            b_predicted_labels_list.append(np.where(df_dev[f'pred_{t["t_name"]}'] == 16, 1, 0))
        
        # Multiclass Confusion Matrix
        true_labels = df_dev['y']
        for i, (predicted_labels, t) in enumerate(zip(predicted_labels_list, thresholds), start=1):
            cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(18)))
            cm_df = pd.DataFrame(cm, index=list(range(18)), columns=list(range(18)))

            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized_df = pd.DataFrame(cm_normalized, index=list(range(18)), columns=list(range(18)))

            html_report += print_confusion_matrices(cm, cm_normalized_df, cm_df, t)
            
            accuracy_multiclass = accuracy_score(true_labels, predicted_labels)
            precision_multiclass = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
            recall_multiclass = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
            f1_multiclass = f1_score(true_labels, predicted_labels, average=None, zero_division=0)

            false_negatives = cm.sum(axis=1) - np.diag(cm)
            true_positives = np.diag(cm)
            false_negative_rate = false_negatives.sum() / (false_negatives.sum() + true_positives.sum())

            results_multiclass = {
                'Model': pipeline.steps[1][0],
                'Classification': 'Multiclass',
                'Num_dim': dimension['num_dim'],
                'Threshold': t['t_value'],
                'Accuracy': round(accuracy_multiclass, 2),
                'Precision': round(precision_multiclass.mean(), 2),
                'Recall': round(recall_multiclass.mean(), 2),
                'F1-score': round(f1_multiclass.mean(), 2), 
                'FN_rate': round(false_negative_rate, 2),
                'Class_Precision': [(i, round(score, 2)) for i, score in enumerate(precision_multiclass)],  
                'Class_Recall': [(i, round(score, 2)) for i, score in enumerate(recall_multiclass)],     
                'Class_F1-score': [(i, round(score, 2)) for i, score in enumerate(f1_multiclass)],
            }
            results.append(results_multiclass)

        # Binary Confusion Matrix
        b_true_labels = np.where(df_dev['y'] == 16, 1, 0)
        for i, (predicted_labels, t) in enumerate(zip(b_predicted_labels_list, thresholds), start=1):
            cm = confusion_matrix(b_true_labels, predicted_labels, labels=list(range(2)))
            cm_df = pd.DataFrame(cm, index=list(range(2)), columns=list(range(2)))

            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized_df = pd.DataFrame(cm_normalized, index=[0,1], columns=[0,1])

            html_report += print_confusion_matrices(cm, cm_normalized_df, cm_df, t)
            
            accuracy_binary = accuracy_score(b_true_labels, predicted_labels)
            precision_binary = precision_score(b_true_labels, predicted_labels, average=None, zero_division=0)
            recall_binary = recall_score(b_true_labels, predicted_labels, average=None, zero_division=0)
            f1_binary = f1_score(b_true_labels, predicted_labels, average=None, zero_division=0)

            false_negatives = cm[1, 0]
            true_positives = cm[1, 1]
            false_negative_rate = false_negatives / (false_negatives + true_positives)

            results_binary = {
                'Model': pipeline.steps[1][0],
                'Classification': 'Binary',
                'Num_dim': dimension['num_dim'],
                'Threshold': t['t_value'],
                'Accuracy': round(accuracy_binary, 2),
                'Precision': round(precision_binary.mean(), 2),
                'Recall': round(recall_binary.mean(), 2),
                'F1-score': round(f1_binary.mean(), 2), 
                'FN_rate': round(false_negative_rate, 2),
                'Class_Precision': [(i, round(score, 2)) for i, score in enumerate(precision_binary)],
                'Class_Recall': [(i, round(score, 2)) for i, score in enumerate(recall_binary)],
                'Class_F1-score': [(i, round(score, 2)) for i, score in enumerate(f1_binary)],
            }
            results.append(results_binary)
    
    html_report += "</body></html>"
    
    # Save the report as an HTML file
    with open(f"Evaluation_report_for_{pipeline.steps[1][0]}.html", "w") as f:
        f.write(html_report)

    return results

def test_dimensions(pipeline, X_data, y_data, thresholds):
    '''
    Function that tests the model pipeline's performance for
    different embedding data dimensionalities and different cut-off thresholds
    that must identify and classify unknowns from politicians
    (where the unknown class is not in the training data).
    :param pipeline: Model pipeline (here: SVC or kNN)
    :param X_data: Data dictionary storing the different embeddings for different dimensionalities
    :param y_data: Data dictionary storing the labels for training, development and testing
    :param thresholds: Data dictionary storing the thresholds and their values
    '''
    results = []
    html_report = f'<html><head><title style="color:black;font-family:Palatino;text-align:center;"> Model Test Report for {pipeline.steps[1][0]}</title></head><body>'
    
    for dimension in X_data:
        predicted_labels_list = []
        b_predicted_labels_list = []
        html_report += f'<h2 style="color:black;font-family:Palatino;text-align:center;">Testing for {dimension["num_dim"]} dimensions</h2>'
        
        # Prepare data infrastructure
        X_train = dimension['X_data']['X_train_final']
        X_test = dimension['X_data']['X_test']

        y_train = y_data[0]['y_train_final']
        y_test = y_data[0]['y_test']

        # Fit the training data to the pipeline
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)
        max_prob_info = [max_probability_info(row) for row in probabilities]
        pred_class, max_prob = zip(*max_prob_info)

        # Store the data in a dataframe for easy access
        df_test = pd.DataFrame({
            'X': X_test.tolist(),
            'y': y_test.tolist(),
            'max_prob': max_prob,
            'pred': pred_class
        })
        df_test['pred'] = df_test['pred'].apply(lambda x: 17 if x == 16 else x) # correction for missing Unknown class in the training data

        # Inspect probabilities
        known_probs = df_test[df_test['y'] != 16]['max_prob']
        unknown_probs = df_test[df_test['y'] == 16]['max_prob']
        html_report += plot_known_unknown_probability_dist(known_probs, unknown_probs, dimension['num_dim'])

        # Recalculate classes for different thresholds
        for t in thresholds:
            df_test[f'pred_{t["t_name"]}'] = df_test.apply(lambda row: 16 if row['max_prob'] < t['t_value'] else row['pred'], axis=1)
            predicted_labels_list.append(df_test[f'pred_{t["t_name"]}'])
            b_predicted_labels_list.append(np.where(df_test[f'pred_{t["t_name"]}'] == 16, 1, 0))
        
        # Multiclass Confusion Matrix
        true_labels = df_test['y']
        for i, (predicted_labels, t) in enumerate(zip(predicted_labels_list, thresholds), start=1):
            cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(18)))
            cm_df = pd.DataFrame(cm, index=list(range(18)), columns=list(range(18)))

            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized_df = pd.DataFrame(cm_normalized, index=list(range(18)), columns=list(range(18)))

            html_report += print_confusion_matrices(cm, cm_normalized_df, cm_df, t)
            
            accuracy_multiclass = accuracy_score(true_labels, predicted_labels)
            precision_multiclass = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
            recall_multiclass = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
            f1_multiclass = f1_score(true_labels, predicted_labels, average=None, zero_division=0)

            false_negatives = cm.sum(axis=1) - np.diag(cm)
            true_positives = np.diag(cm)
            false_negative_rate = false_negatives.sum() / (false_negatives.sum() + true_positives.sum())

            results_multiclass = {
                'Model': pipeline.steps[1][0],
                'Classification': 'Multiclass',
                'Num_dim': dimension['num_dim'],
                'Threshold': t['t_value'],
                'Accuracy': round(accuracy_multiclass, 2),
                'Precision': round(precision_multiclass.mean(), 2),
                'Recall': round(recall_multiclass.mean(), 2),
                'F1-score': round(f1_multiclass.mean(), 2), 
                'FN_rate': round(false_negative_rate, 2),
                'Class_Precision': [(i, round(score, 2)) for i, score in enumerate(precision_multiclass)],  
                'Class_Recall': [(i, round(score, 2)) for i, score in enumerate(recall_multiclass)],     
                'Class_F1-score': [(i, round(score, 2)) for i, score in enumerate(f1_multiclass)],
            }
            results.append(results_multiclass)

        # Binary Confusion Matrix
        b_true_labels = np.where(df_test['y'] == 16, 1, 0)
        for i, (predicted_labels, t) in enumerate(zip(b_predicted_labels_list, thresholds), start=1):
            cm = confusion_matrix(b_true_labels, predicted_labels, labels=list(range(2)))
            cm_df = pd.DataFrame(cm, index=list(range(2)), columns=list(range(2)))

            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized_df = pd.DataFrame(cm_normalized, index=[0,1], columns=[0,1])

            html_report += print_confusion_matrices(cm, cm_normalized_df, cm_df, t)
            
            accuracy_binary = accuracy_score(b_true_labels, predicted_labels)
            precision_binary = precision_score(b_true_labels, predicted_labels, average=None, zero_division=0)
            recall_binary = recall_score(b_true_labels, predicted_labels, average=None, zero_division=0)
            f1_binary = f1_score(b_true_labels, predicted_labels, average=None, zero_division=0)

            false_negatives = cm[1, 0]
            true_positives = cm[1, 1]
            false_negative_rate = false_negatives / (false_negatives + true_positives)

            results_binary = {
                'Model': pipeline.steps[1][0],
                'Classification': 'Binary',
                'Num_dim': dimension['num_dim'],
                'Threshold': t['t_value'],
                'Accuracy': round(accuracy_binary, 2),
                'Precision': round(precision_binary.mean(), 2),
                'Recall': round(recall_binary.mean(), 2),
                'F1-score': round(f1_binary.mean(), 2), 
                'FN_rate': round(false_negative_rate, 2),
                'Class_Precision': [(i, round(score, 2)) for i, score in enumerate(precision_binary)],
                'Class_Recall': [(i, round(score, 2)) for i, score in enumerate(recall_binary)],
                'Class_F1-score': [(i, round(score, 2)) for i, score in enumerate(f1_binary)],
            }
            results.append(results_binary)
    
    html_report += "</body></html>"
    
    # Save the report as an HTML file
    with open(f"Test_report_for_{pipeline.steps[1][0]}.html", "w") as f:
        f.write(html_report)

    return results

# Plot the different development metrics
def plot_metrics(data, title):
    '''
    Function to visualize the performance behavior for the different models in different dimensions
    :param data: The model that is being visualized
    :param title: The title that belongs to the model
    '''
    unique_dims = data['Num_dim'].unique()
    num_plots = len(unique_dims)
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 6, 5))
    fig.suptitle(title, fontfamily='Palatino', fontsize=25)

    if num_plots == 1:
        axes = [axes]

    # Calculate global min and max for y-axis limits
    y_min = 0
    y_max = 1.1

    for i, dim in enumerate(unique_dims):
        ax = axes[i]
        subset = data[data['Num_dim'] == dim]
        ax.plot(subset['Threshold'], subset['Accuracy'], label='Accuracy', marker='o')
        ax.plot(subset['Threshold'], subset['Precision'], label='Precision', marker='o')
        ax.plot(subset['Threshold'], subset['Recall'], label='Recall', marker='o')
        ax.plot(subset['Threshold'], subset['F1-score'], label='F1-score', marker='o')

        ax.set_title(f'Dimension: {dim}', fontdict=title_font)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metrics')
        ax.set_ylim([y_min, y_max])
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def save_face(image, face, count, label, output_dir):
    '''
    Writes the unique faces found in detect_cut_save_faces(_test) to file
    :param image: url where image is stored online (no download required)
    :param face: Bounding box indicating the location of the face in the image
    :param count: Index variable
    :param label: Data-dependent identifier for an image (here: a politician's name or news artice id)
    :param output_dir: Directory where images are stored
    '''
    x, y, w, h = face
    face_img = image[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_img)
    
    # Check if subfolders are needed in case of dealing with labelled data
    if not any(char.isdigit() for char in label):
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        face_pil.save(os.path.join(label_dir, f'{label}_{count:03d}.jpg'))
    else:
        face_pil.save(os.path.join(output_dir, f'{label}_{count:03d}.jpg'))

def detect_cut_save_faces(df, url_column, label_column, output_dir):
    '''
    Function to detect, cut and save faces from the img_urls opened online 
    using both HAAR cascades and the built-in face_recognition library
    
    :param url_column: Reference to the column that stores the image url to be scraped
    :param label_column: Binary reference to either politician (for training/testing data) or article id (for news image data)
    :param output_dir: Directory where images are stored
    '''
    face_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Scraping at..."):
        url = row[url_column]
        label = str(row[label_column])

        if not url:
        # Skip if the URL is empty
            continue
        
        try:
            # Download the image
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
        except (requests.RequestException, UnidentifiedImageError) as e:
            continue

        if len(img.shape) != 3 or img.shape[2] != 3:
            print(f"Image at {url} does not have 3 channels.")
            continue
        
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        haar_faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Detect faces using face_recognition
        face_locations = face_recognition.face_locations(img)
        fr_faces = [(left, top, right-left, bottom-top) for top, right, bottom, left in face_locations]
        
        # Combine and filter faces based on overlap
        all_faces = list(haar_faces) + fr_faces
        
        unique_faces = []
        for face in all_faces:
            if not any(overlap_ratio(face, uf) > 0.5 for uf in unique_faces):
                unique_faces.append(face)
      
        # Write unique faces to file
        for face in unique_faces:
            save_face(img, face, face_count, label, output_dir)
            face_count += 1