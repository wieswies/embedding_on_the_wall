# For necessary data processing and calculations
import numpy as np
import pandas as pd

# Regular expressions
import re
import ast

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

# For image download
import requests
from PIL import Image

# For data visualization
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize
import plotly.graph_objs as go
from plotly.offline import plot
from itertools import combinations

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

def model_name(x):
    '''
    Custom lambda function that maps the model key to a displayable name
    '''
    if x == 'svc_rbf':
        return 'SVM (rbf)'
    elif x == 'knn_10':
        return 'kNN (k=10)'
    elif x == 'knn_20':
        return 'kNN (k=20)'
    elif x == 'knn_30':
        return 'kNN (k=30)'

def compute_confusion_matrix(df, items_list, col1, col2):
    '''
    Function to compute a confusion matrix listing co-occurrences.
    Here applied to news articles
    :param df: dataframe with the news articles by id and classifications for news images, 
    per item (i.e. one article id can have multiple entries, one entry per item)
    :param items_list: items, stored in a list in the desired order of display
    :param col1: identifier column (here: article id)
    :param col2: item column, storing the items of which one wants to compute the co-occurrence
    '''
    grouped = df.groupby(col1)[col2].apply(lambda x: list(set(x)))
    co_occurrences = []
    for items in grouped:
        if len(items) > 1:
            co_occurrences.extend(combinations(items, 2))
    co_occurrence_counts = Counter(co_occurrences)

    confusion_matrix = pd.DataFrame(0, index=items_list, columns=items_list)
    for (item1, item2), count in co_occurrence_counts.items():
        confusion_matrix.loc[item1, item2] = count
        confusion_matrix.loc[item2, item1] = count

    return confusion_matrix