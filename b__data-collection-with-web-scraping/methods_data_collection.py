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

# For image processing
from PIL import Image, UnidentifiedImageError
import face_recognition
import dlib
import cv2

# For face-embedding calculation
from keras_facenet import FaceNet as FN

# For machine learning
import tensorflow as tf

# For web scraping
import html
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# For tracking programming progress
from tqdm.notebook import tqdm
import time

def parse_nu_media_data(val):
    '''
    Function to parse .JSON stored as object datatype in pandas column (here applied to NUnl media)
    '''
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (ValueError, json.JSONDecodeError):
            return None
    return val

# HAAR cascade variables for face detection
haar_cascade_path = 'haarcascade_frontalface_default.xml'
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_cascade_path)

def overlap_ratio(box1, box2):
    '''
    Function to calculate the overlap between two face-bounding-boxes detected in an image
    :param box1: First detected face
    :param box2: Second detected face
    '''
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


def detect_cut_save_faces_test(df, url_column, label_column, output_dir):
    '''
    Function to detect, cut and save faces from the img_urls opened online 
    using both HAAR cascades and the built-in face_recognition library
    Includes print statements to inspect accuracy
    
    :param url_column: Reference to the column that stores the image url to be scraped
    :param label_column: Binary reference to either politician (for training/testing data) or article id (for news image data)
    :param output_dir: Directory where images are stored
    '''
    face_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Test scraping at..."):
        url = row[url_column]
        label = str(row[label_column])
        print(f'{label} with image url: {url}')

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
        
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        haar_faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f'#HAAR: {len(haar_faces)} at: {haar_faces}')
        
        # Detect faces using face_recognition
        face_locations = face_recognition.face_locations(img)
        fr_faces = [(left, top, right-left, bottom-top) for top, right, bottom, left in face_locations]
        print(f'#FR: {len(fr_faces)} at: {fr_faces}')
        
        # Combine and filter faces based on overlap
        all_faces = list(haar_faces) + fr_faces
        print(f'#COMBINED: {len(all_faces)}')

        
        unique_faces = []
        for face in all_faces:
            if not any(overlap_ratio(face, uf) > 0.5 for uf in unique_faces):
                unique_faces.append(face)
        print(f'#UNIQUE: {len(unique_faces)} at: {unique_faces}\n')
      
        # Write unique faces to file
        for face in unique_faces:
            save_face(img, face, face_count, label, output_dir)
            face_count += 1

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

def get_image_dataset_embeddings(data_folder, dataset, embedder):
    '''
    Function that 1) loops through the data folder that stores facial images 
    that were detected by applying HAAR cascades and face_detection 
    and 2) calculates embeddings with the keras-facenet wrapper implementation
    :param data_folder: Data folder that stores cut-outs of facial images
    :param dataset: Label that the image belongs to (here either NOS.nl images, NU.nl images, or polician faces
    :param embedder: Embedder applied (here: Facenet, see also https://www.cvfoundation.org/
    openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
    '''
    X = []
    y = []
    z = []
    p = []
    images_to_process = []
    image_paths = []
    
    # 1: Collect all images
    for iso_news_photo in tqdm(os.listdir(data_folder), desc='Preparing images'):
        if not iso_news_photo.endswith('.jpg'):
            continue
        
        iso_news_photo_path = os.path.join(data_folder, iso_news_photo)
        image = cv2.imread(iso_news_photo_path)
        
        if image is not None:
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_to_process.append(image_rgb)
                image_paths.append(iso_news_photo_path)
            except Exception as e:
                print(f"Error processing image {iso_news_photo_path}: {e}")
        else:
            print(f"Warning: Unable to read image {iso_news_photo_path}")
    
    # 2: Extract embeddings in batch
    try:
        print("Extracting embeddings...")
        embeddings = embedder.embeddings(images_to_process)  
        print("Embeddings extracted.")
        
        # Iterate through the results and store them
        for embedding, path in zip(embeddings, image_paths):
            X.append(embedding)
            y.append(dataset)
            z.append(os.path.basename(path))
            p.append(path)
    
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
    
    return X, y, z, p