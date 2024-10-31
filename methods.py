# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import shutil
import time
from tqdm import tqdm

import requests
import cv2
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import face_recognition
import dlib

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

# Lists
# %%
sports_categories = [
    'Alpineskiën',
    'American football',
    'Atletiek',
    'Autosport',
    'BMX',
    'Baanwielrennen',
    'Badminton',
    'Basketbal',
    'Beachvolleybal',
    'Biljarten',
    'Boksen',
    'Cricket',
    'Curling',
    'Dammen',
    'Darts',
    'Futsal',
    'Golf',
    'Handbal',
    'Hockey',
    'IJshockey',
    'Judo',
    'Kanovaren',
    'Korfbal',
    'Kunstschaatsen',
    'Marathonschaatsen',
    'Motocross',
    'Motorsport',
    'Mountainbiken',
    'Paardensport',
    'Paardensport dressuur',
    'Paardensport eventing',
    'Paardensport springen',
    'Paralympische sport',
    'Roeien',
    'Rugby',
    'Schaatsen',
    'Schaken',
    'Shorttrack',
    'Skateboarden',
    'Skeleton',
    'Skispringen',
    'Snooker',
    'Snowboarden',
    'Sport algemeen',
    'Tafeltennis',
    'Tennis',
    'Triatlon',
    'Turnen',
    'Veldrijden',
    'Voetbal',
    'Waterpolo',
    'Zeilen',
    'Zwemmen'
    'Voetbal',
]

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
    # Class used during embedding calculation to prevent time-out by overload in print statements
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        pass
        
# %%
# Function to calculate overlap ratio to not have duplicates as a result of two face detection techniques
def overlap_ratio(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    overlap = inter_area / float(box1_area + box2_area - inter_area)
    return overlap

# Function to cut out and save the face
def save_face(image, face, count, label):
    x, y, w, h = face
    face_img = image[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_img)
    # Create label directory if it doesn't exist
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    face_pil.save(os.path.join(label_dir, f'face_{count}.jpg'))

# Function to process the dataframe
def scrape_isolated_traintest_faces(df):
    face_count = 0
    
    for index, row in df.iterrows():
        url = row['img_link']
        label = row['politician']

        try:
            # Download the image
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
        except (requests.RequestException, UnidentifiedImageError) as e:
            print(f"Error downloading or opening image from URL {url}: {e}")
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
      
        # Save unique faces
        for face in unique_faces:
            save_face(img, face, face_count, label)
            face_count += 1

# %%
# Method to extract the face embedding of an image
def get_face_embedding(face_image):
    detections = embedder.extract(face_image, threshold=0.95)
    
    # Check if a face was found
    if len(detections) > 0:
        # Return the embedding of the found face
        return detections[0]['embedding']
    else: return None

# %%

