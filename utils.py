import os
import glob
import re
import string
import csv
import contractions
import spacy  # Import spaCy
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import community as community_louvain

# 📌 Fix: Load spaCy model globally
nlp = spacy.load("en_core_web_sm")
# Load spaCy model and NLP tools
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))




# Load proper nouns from a CSV file
def load_proper_nouns(file_path):
    proper_nouns = set()
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return proper_nouns  # Return an empty set if the file isn't found
    
    # Open the file with the correct encoding
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) > 0:  # Check if the row is not empty
                proper_nouns.add(row[0].strip().lower())  # Add each word from the first column
    
    return proper_nouns

def preprocess_text(text):
    # Replace newlines with spaces and remove non-alphanumeric characters
    text = text.replace('\n', ' ')
    text = text.lower()

    # Expand contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text)
    
    tokens = nltk.tokenize.word_tokenize(text, language='english')
    tokens = [word for word in tokens if word.isalnum()]
    # Remove tokens that are only numeric
    tokens = [word for word in tokens if not word.isdigit()]
    # Remove tokens that contain both letters and numbers
    tokens = [word for word in tokens if not re.search(r'\d', word)]
    return tokens



def filter_text(text, proper_nouns, stop_words):
    text = preprocess_text(text)
    doc = nlp(' '.join(text))  # Create spaCy doc from tokens
    words = [
        token.lemma_ for token in doc
        if token.pos_ in ['NOUN', 'VERB', 'ADJ']  # Nouns, Verbs, Adjectives
        and token.lemma_ not in stop_words
        and token.lemma_ not in proper_nouns
        and len(token.lemma_) > 2  # Minimum word length
    ]
    
    return words


def filter_propernouns(tokens, proper_nouns):
    return [token for token in tokens if token not in proper_nouns]

# Load the word lists for frequent words (headwords) and convert to lowercase
def load_word_list(file_path):
    """
    Load a list of words from a file and transform them to lowercase.
    Expects a fully resolved file path.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return set()  # Return an empty set if file is missing

    with open(file_path, "r", encoding="utf-8") as file:
        return set(word.strip().lower() for word in file)  # Convert each word to lowercase


def process_text_files(directory, book_ranges, proper_nouns, stop_words):
    word_counts = Counter()
    word_locations = defaultdict(set)  # Using set to avoid duplicates
    
    for book, units in book_ranges.items():
        for unit in units:
            filename = f"{book}_{unit}_More.txt"
            filepath = os.path.join(directory, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    if not text.strip():  # ✅ Skip empty files
                        print(f"Warning: Empty file skipped - {filepath}")
                        continue

                    tokens = filter_text(text, proper_nouns, stop_words)
                    
                    for token in tokens:
                        word_counts[token] += 1
                        word_locations[token].add(f"Book{book}_Unit{unit}")
    
    return word_counts, word_locations
    

# Now define a function to map the original tokens to their headwords
def map_to_headwords(tokens):
    """
    Map tokens (words) to their headwords using lemmatization.
    Keeps track of the original form as well.
    Returns a dictionary: {original_word: headword}.
    """
    doc = nlp(' '.join(tokens))
    return {token.text: token.lemma_ for token in doc}  # Map original word to its headword


# Preprocess the lists by lemmatizing the words, so they match the processed tokens
def lemmatize_word_list(word_list):
    """
    Lemmatizes the words in the word list using spaCy.
    """
    return set(token.lemma_ for token in nlp(' '.join(word_list)))


def read_text_files(directory):
    """
    Reads all text files from the specified directory.
    Returns a dictionary where keys are filenames and values are text content.
    """
    pattern = "*.txt"
    file_paths = glob.glob(os.path.join(directory, pattern))

    texts = {}
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                texts[os.path.basename(file_path)] = file.read()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return texts

# Compute word frequencies across all texts
def compute_word_frequencies(texts, proper_nouns, stop_words):
    """
    Compute the frequency of words across all texts after preprocessing and filtering.
    Returns a Counter object with word frequencies.
    """
    all_words = []
    for text in texts.values():
        words = filter_text(text, proper_nouns, stop_words)
        all_words.extend(words)  # Add all words from each document to the global list

    word_frequencies = Counter(all_words)
    return word_frequencies

# Step 5: Compute TF-IDF scores
def compute_tfidf(texts, min_tfidf_threshold=0.01):
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()

    filtered_words_by_doc = []
    for doc_scores in tfidf_scores:
        filtered_words = [word for word, score in zip(words, doc_scores) if score >= min_tfidf_threshold]
        filtered_words_by_doc.append(filtered_words)
    return filtered_words_by_doc



def build_cooccurrence_matrix(words, window_size):
    unique_words = list(set(words))  # Get unique words
    word_to_index = {word: idx for idx, word in enumerate(unique_words)}

    # Initialize adjacency matrix
    adj_matrix = np.zeros((len(unique_words), len(unique_words)))

    # Create co-occurrence within the specified window size
    for i in range(len(words)):
        for j in range(i + 1, min(i + window_size, len(words))):
            word_i = words[i]
            word_j = words[j]
            if word_i in word_to_index and word_j in word_to_index:
                idx_i = word_to_index[word_i]
                idx_j = word_to_index[word_j]
                adj_matrix[idx_i][idx_j] += 1
                adj_matrix[idx_j][idx_i] += 1  # Symmetric matrix

    return adj_matrix, unique_words

def order_dataframe(df):
    # Sum rows and columns
    row_sums = df.sum(axis=1)
    col_sums = df.sum(axis=0)
    
    # Order indices by sums
    ordered_row_indices = row_sums.sort_values(ascending=False).index
    ordered_col_indices = col_sums.sort_values(ascending=False).index
    
    # Reindex DataFrame
    df = df.loc[ordered_row_indices, ordered_col_indices]
    
    return df


# Function to get edge list from adjacency matrix DataFrame
def adjacency_matrix_df_to_edge_list(df, interval_label=None):
    edge_list = []
    
    # Iterate over the DataFrame
    for i, row in df.iterrows():
        for j, value in row.items():
            if value != 0:  # If there's an edge
                edge_list.append((i, j, value, interval_label))  # Add edge (i, j) with weight value and interval label
    
    return edge_list

def assign_node_shape(node, list1_words, list2_words, list3_words):
    """
    Assigns a shape to the node (word) based on the list it belongs to.
    Circle: list1, Triangle: list2, Square: list3, Deltoid: not in any list.
    """
    if node in list1_words:
        return 'circle'
    elif node in list2_words:
        return 'triangle'
    elif node in list3_words:
        return 'square'
    else:
        return 'deltoid'  # Not in any list

def save_cooccurrence_matrix(directory, matrix, words, filename="cooccurrence_matrix.csv"):
    """
    Saves a co-occurrence matrix as a CSV file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Ensure matrix is a DataFrame before saving
    df = pd.DataFrame(matrix, index=words, columns=words)
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path)

    print(f"Co-occurrence matrix saved at {file_path}")


def save_output_file(directory, filename, content):
    """
    Saves text content to a file inside a specified directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create directory if it doesn't exist

    file_path = os.path.join(directory, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"File saved as {file_path}")




