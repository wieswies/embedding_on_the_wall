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

def list_occurrences_triple_column(dataframe, colnames_of_interest, wordlists):
    '''
    Function that finds words that relate to topics. Here, the topics are the wordlist identifiers and mainly include political parties. 
    :param dataframe: dataframe that stores the text columns that need to be scanned
    :param colnames_of_interest: columns that store text. Here, these are the title, paragraphs and alt_text that were scraped from a news website
    :wordlists: a dictionary that stores wordlists by identifier, i.e. topic. Given that we are interested in multistrings, 
                substrings and exact string  matches, these are all looped through in this function for greater reach and accuracy
    '''

    occurrences_df = dataframe.copy()

    for colname in colnames_of_interest:
        dataframe[colname] = dataframe[colname].astype(str)
        
        for identifier, wordlist_options in wordlists.items():
            all_matches = [[] for _ in range(len(occurrences_df))]
            
            for wordlist_option, wordlist in wordlist_options.items():
                
                if wordlist_option == 'multistring_matches':
                    for i, row in dataframe.iterrows():
                        
                        multistring_matches = []
                        for word in wordlist:
                            word_lower = word.lower()
                            matches = [m.start() for m in re.finditer(r'\b' + re.escape(word_lower) + r'\b',
                                                                      row[colname].lower())]
                            multistring_matches.extend([word] * len(matches))
                        
                        all_matches[i].extend(multistring_matches)

                if wordlist_option == 'substring_matches':
                    for i, row in dataframe.iterrows():

                        words = re.findall(r'\w+', row[colname])
                        substring_matches = [ss for ss in words if
                                             any(sub.lower() in ss.lower() for sub in wordlist)]
                        all_matches[i].extend(substring_matches)
    
                elif wordlist_option == 'exact_matches':
                    for i, row in dataframe.iterrows():
                        words = re.findall(r'\w+', row[colname])
                        exact_matches = [em for em in words if
                                         any(sub.lower() == em.lower() for sub in wordlist)]
                        all_matches[i].extend(exact_matches)
    
            occurrences_df[f'{identifier}_{colname}'] = all_matches
    return occurrences_df

def remove_words(input, word_list):
    '''
    Removes words that are in the word_list from the input
    '''
    words_to_remove = []
    for word in word_list:
        for occurrence in input:
            if word == occurrence:
                words_to_remove.append(word)
    for word in words_to_remove:
        input.remove(word)
    return input

def replace_words(input, search_list, mapping):
    return [mapping if x in search_list else x for x in input]
    
def remove_words_from_colnames(dataframe, colnames, wordlist):
    '''
    Method that removes words from specified columns in a dataframe based on wordlists, while storing those words in a new dataframe
    :param dataframe: dataframe from which words are to be removed
    :param colnames: colnames of interest
    :param wordlist: wordlist containing the word matches that should be removed
    Returns: dataframe with words removed (updated_words_df), and a dataframe in which the removed words are stored (removed_words_df)
    '''
    # Create two dataframes eventually constituting the return statement
    updated_words_df = dataframe.copy()
    removed_words_df = dataframe.copy()

    # Loop over columns of interest
    for col in colnames:
        # Loop over the content
        for index, row in dataframe.iterrows():
            # Split words to be removed from input and store them in a new list
            updated_col = [word for word in row[col] if word not in wordlist]
            removed_col = [word for word in row[col] if word in wordlist]

            # Store the list with the words without from the wordlist in updated_words_df,
            # store the list with the words that were removed in removed_words_df
            updated_words_df.at[index, col] = updated_col
            removed_words_df.at[index, col] = removed_col

    # Return both dataframes
    return updated_words_df, removed_words_df

def conditional_remove_and_map_words(input, search_list, mapping):
    '''
    Method that 1) splits the input in modified_input and stripped_input, 2) removes the substrings of the stripped_input
    from the modified_input, and 3) adds a mapping of the stripped_input to the modified_input.

    Example input: ['GroenLinks/PvdA', 'GroenLinks', 'PvdA', 'PvdA', 'Timmermans']
    -- this means that by design of the list_occurrences_double_column method, all substrings of the multistring are duplicates --
    modified_input = ['GroenLinks', 'PvdA', 'PvdA', 'Timmermans]
    stripped_input = ['GroenLinks/PvdA']
    returns: ['PvdA', 'Timmermans', 'GL-PvdA']

    :param input: dataframe cell containing a wordlist (to be applied in lambda statement)
    :param search_list: list of possible ways to reference a multistring
    :param mapping: mapping that should replace any of the references from the search_list with a consistent mapping
    :return: modified_list where substring duplicates are removed and the references to the multi-string are made consistent
    '''

    # Split the input in modified_input and stripped_input, where stripped input contains possible references to a multi-string
    modified_input = [i for i in input if i not in search_list]
    stripped_input = [i for i in input if i in search_list]

    # Loop over the multistring references that appeared in the search_list
    for word in stripped_input:
        # Split the multistring in substrings
        subs = re.split(r'[-/]', word)

        # Remove these substrings from the modified input, since they are duplicates
        for sub in subs:
            if sub in modified_input:
                modified_input.remove(sub)

        # Append a consistent mapping of the multistring
        modified_input.append(mapping)
    return modified_input

def count_colname_presence(df, colname_list):
    colname_presence_count = dict()
    for colname in colname_list:
        colname_presence_count[colname] = df[colname].apply(lambda x: 1 if x else 0).sum()
    return colname_presence_count

def check_words_list(pattern, lst):
    for text in lst:
        if re.search(pattern, text):
            return True
    return False

def download_img_from_url(folder, dataframe, id_column, clf_column, url_column):
    '''
    Function that takes the url and downloads the image with the classification as filename.
    :param folder: folder to which image must be written
    :param dataframe: dataframe in which the image urls are stored
    :param id_column: column that holds the unique identifer of the item, to be added to the filename
    :param clf_column: column that holds the lists with classifications, to be added to the filename for easy checking
    :param url_column: column that holds the url information, which need to be opened and scraped
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    for i, row in dataframe.iterrows():
        url = row[url_column]
        filename = f'{row[id_column]}__{row[clf_column]}.jpg'
        file_path = os.path.join(folder, filename) 
    
        try:
            image_content = requests.get(url, timeout=10).content
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file)
    
            with open(file_path, "wb") as f:
                image.save(f, "JPEG")
    
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}")

def is_non_empty_list(value):
    return isinstance(value, list) and len(value) > 0

def is_list_and_non_empty(value):
    if isinstance(value, str):
        value = value.strip()
        if value == '[]':
            value = []
        elif value.startswith('[') and value.endswith(']'):
            try:
                value = eval(value)
                if not isinstance(value, list):
                    value = []
            except:
                value = []
        else:
            value = []
    elif not isinstance(value, list):
        value = [value]
    
    return isinstance(value, list) and len(value) > 0

def fix_and_eval_list(value):
    '''
    Function to be applied as a lambda function that restores the datatype(list)
    '''
    if pd.isna(value): 
        return []
        
    if isinstance(value, list):
        return sorted(value)
        
    try:
        value_fixed = value.replace("' '", "', '").replace("'   '", "', '").replace("''", "', '")
        new_list = ast.literal_eval(value_fixed)
        return sorted(new_list)
    except (ValueError, SyntaxError):
        return value

def compare_lists_as_sets(row, true_clf, uncorrected_clf):
    '''
    Compares two lists, by converting them to sets and returning the set difference. Here applied to the uncorrected and corrected classifications, to find out which classifications were missed by the face detector at what frequency.
    :param row: row to be applied to (lambda function)
    :param true_clf: column that stores the lists with true classfications
    :param uncorrected_clf: column that stores the lists with true classfications before the correction, or in other words, the corrections that are not complete
    '''
    set1 = set(row[true_clf]) if isinstance(row[true_clf], list) else set()
    set2 = set(row[uncorrected_clf]) if isinstance(row[uncorrected_clf], list) else set()
    
    return list(set1.difference(set2))

def check_clf_ref(pol_img_df, dict):
    '''
    Customized function that checks whether a politician or their party is referenced in the title, specifically for those politicians appearing in the image
    :param pol_img_df: dataframe that stores article data for articles where a politician appeared in the main image
    :param dict: dictionary that stores which politician belongs to which party
    '''
    pol_ref_results = []
    for i, row in pol_img_df.iterrows():
        for clf in row['politician_in_img']:
            party = dict[clf]
            column_to_check = f'{party}_title'
            value = is_list_and_non_empty(row[column_to_check])
            pol_ref = {
                'dataset': row['dataset'],
                'id': row['id'],
                'url': row['url'],
                'title': row['title'],
                'politician_in_img': row['politician_in_img'],
                'per_politician': clf,
                'ref_in_title': value,
                'ref': row[column_to_check]
            }
            pol_ref_results.append(pol_ref)
    return pd.DataFrame(pol_ref_results)