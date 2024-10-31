# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import shutil
import time
from tqdm import tqdm

import cv2
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from keras_facenet import FaceNet

# Methods
# %%
def delete_ds_store_files(directory):
    '''
    Method to remove the .DS_Store files that unnoticeably appear in the directory
    :param directory: directory from which to remove the .DS_Store files
    :return: print statements on which files were removed
    '''
    for ds_store_file in directory.rglob(".DS_Store"):
        try:
            if ds_store_file.is_file():
                ds_store_file.unlink()  # Delete the file
                print(f"Deleted: {ds_store_file}")
            else:
                print(f"Skipping: {ds_store_file} (not a file)")
        except Exception as e:
            print(f"Error deleting {ds_store_file}: {e}")

# %%
class HiddenPrints:
    '''
    Class to hide embeddingprints when running the calculations for FaceNet embeddings
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# %%
def get_face_embedding(embedder, face_image):
    '''
    # Method to extract the face embedding of an image
    :param embedder: the embedder used to extract the face embedding, e.g. FaceNet
    :param face_image: input image to calculate embedding for, given with: cv2.imread(iso_img_path)
    :return: an array with embeddings
    '''
    detections = embedder.extract(face_image, threshold=0.95)
    time.sleep(0.5)
    # Check if a face was found
    if len(detections) > 0:
        # Return the embedding of the found face
        return detections[0]['embedding']
    else:
        return None

# %%
def store_face_embedding(embedder, data_folder):
    '''
    Runs the embedding calculations with FaceNet and stores these embeddings for all files in X, together with
    their corresponding labels (y), filenames containing the article id (z), and in case the embedding is empty,
    this file is stored in empty_embeddings
    :param iteration: initialized with 0
    :param data_folder: the directory in which the folders with faces are stored
    :return: X, y, z, empty_embeddings
    '''
    X, y, z, empty_embeddings = [], [], [], []
    for politician_subfolder in tqdm(os.listdir(data_folder), desc='Embedding, embedding, embedding aan de wand',
                                     unit='politician_subfolder'):
        politician_path = os.path.join(data_folder, politician_subfolder)

        # Loop over the image files in the directory
        for iso_img_file in os.listdir(politician_path):

            # Check if file is an image
            if not iso_img_file.endswith('.jpeg'):
                continue
            iso_img_path = os.path.join(politician_path, iso_img_file)
            iso_img = cv2.imread(iso_img_path)

            # Read the image files and extract the face_embeddings with cv2
            if iso_img is not None:
                with HiddenPrints():
                    # Call method to extract the face embedding
                    '''face_embedding = get_face_embedding(embedder, cv2.imread(iso_img_path))'''
                    detections = embedder.extract(iso_img, threshold=0.95)
                    time.sleep(0.5)
                    # Check if a face was found
                    if len(detections) > 0:
                        # Return the embedding of the found face
                        face_embedding = detections[0]['embedding']
                        if face_embedding is not None:
                            X.append(face_embedding)
                            y.append(politician_subfolder)
                            z.append(iso_img_file)
                        else:
                            empty_embeddings.append((politician_subfolder, iso_img_file))

    return X, y, z, empty_embeddings

def store_face_embedding(embedder, data_folder):
    '''
    Runs the embedding calculations with FaceNet and stores these embeddings for all files in X, together with
    their corresponding labels (y), filenames containing the article id (z), and in case the embedding is empty,
    this file is stored in empty_embeddings
    :param iteration: initialized with 0
    :param data_folder: the directory in which the folders with faces are stored
    :return: X, y, z, empty_embeddings
    '''
    X, y, z, empty_embeddings = [], [], [], []
    for politician_subfolder in tqdm(os.listdir(data_folder), desc='Embedding, embedding, embedding aan de wand',
                                     unit='politician_subfolder'):
        politician_path = os.path.join(data_folder, politician_subfolder)
        
        # Loop over the image files in the directory
        for iso_img_file in os.listdir(politician_path):

            # Check if file is an image
            if not iso_img_file.endswith('.jpeg'):
                continue
            iso_img_path = os.path.join(politician_path, iso_img_file)
            iso_img = cv2.imread(iso_img_path)

            # Read the image files and extract the face_embeddings with cv2
            if iso_img is not None:
                with HiddenPrints():
                    # Call method to extract the face embedding
                    '''face_embedding = get_face_embedding(embedder, cv2.imread(iso_img_path))'''
                    detections = embedder.extract(iso_img, threshold=0.95)
                    print(f'Detected embeddings for {politician_subfolder}, picture {iso_img_file}: {detections}')
                    time.sleep(0.5)
                    # Check if a face was found
                    if len(detections) > 0:
                        # Return the embedding of the found face
                        face_embedding = detections[0]['embedding']
                        if face_embedding is not None:
                            X.append(face_embedding)
                            y.append(politician_subfolder)
                            z.append(iso_img_file)
                        else:
                            empty_embeddings.append((politician_subfolder, iso_img_file))
                            
    return X, y, z, empty_embeddings