import re
import nltk
import tarfile
import os
import io
import itertools
import pickle
from typing import List, Tuple, Dict, Any, Optional, Iterator 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def tokenize_and_clean_review(text_review):
    """
    Cleans, tokenizes, and stems a raw movie review.

    Args:
        text_review (str): The raw text of a movie review.

    Returns:
        list: A list of cleaned, tokenized, and stemmed words.
    """
    # Convert text to lowercase
    text_review = text_review.lower()

    # Remove HTML tags if present
    text_review = re.sub(r'<.*?>', '', text_review)

    # Remove punctuation and special characters
    text_review = re.sub(r'[^a-z\s]', '', text_review)

    # Tokenize the text into a list of words
    tokens = word_tokenize(text_review)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stem the words to their roots
    porter = PorterStemmer()
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]

    return stemmed_tokens


def process_reviews_from_archive(
    file_path: str,
    num_to_skip: int = 0,
    limit: Optional[int] = None
) -> Dict[str, List[Any]]:
    """
    Reads, labels, and tokenizes movie reviews from a .tar.gz archive.

    This function expects the archive to have a folder structure where the
    sentiment label (e.g., 'pos' or 'neg') and dataset split (e.g., 'train' or 'test')
    are part of the file path.

    Args:
        file_path (str): The full path to the .tar.gz archive.
        num_to_skip (int): The number of archive members to skip before starting.
        limit (Optional[int]): The maximum number of archive members to process.

    Returns:
        Dict[str, List[Any]]: A dictionary containing the processed reviews,
                             with keys for each dataset split and sentiment.
    """
    results = {
        'all_processed_reviews_train': [],
        'all_processed_reviews_test': [],
        'unsup_processed_reviews': [],
    }

    print(f"Starting to process files from '{file_path}'.")
    print(f"Skipping the first {num_to_skip} files.")
    if limit is not None:
        print(f"Processing a maximum of {limit} files.")

    try:
        with tarfile.open(file_path, "r:gz") as tar:
            members_to_process = itertools.islice(
                tar,
                num_to_skip,
                None if limit is None else num_to_skip + limit
            )
            
            for member in members_to_process:
                # Check for file type
                if member.isfile() and member.name.endswith('.txt') and "urls" not in member.name.lower():
                    member_path_lower = member.name.lower()
                    
                    # Determine sentiment and dataset split
                    sentiment_label = None
                    if 'pos' in member_path_lower:
                        sentiment_label = 'positive'
                    elif 'neg' in member_path_lower:
                        sentiment_label = 'negative'
                    elif 'unsup' in member_path_lower:
                        sentiment_label = 'unsup'
                    else:
                        print("Skipping unneeded file.")
                        continue

                    # Split dataset
                    dataset_split = None
                    if "train" in member_path_lower:
                        dataset_split = "train"
                    elif "test" in member_path_lower:
                        dataset_split = "test"
                    else:
                        print(f"Skipping file with unknown dataset split: {member.name}")
                        continue
                    
                    f = tar.extractfile(member)
                    if f:
                        raw_review = f.read().decode('utf-8')
                        processed_review = tokenize_and_clean_review(raw_review)
                        
                        # Append to the correct list based on sentiment and split
                        if sentiment_label == 'unsup':
                            key = f"unsup_processed_reviews"
                            results[key].append((processed_review, dataset_split))
                        else:
                            key = f"all_processed_reviews_{dataset_split}"
                            results[key].append((processed_review, sentiment_label, dataset_split))

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        print("Please ensure the path is correct and the file exists.")
    except tarfile.ReadError:
        print(f"Error: Could not read the file at '{file_path}'.")
        print("This may not be a valid .tar.gz archive.")
    
    return results


def save_with_pickle(data, filename):
    """
    Saves a Python object to a file using the pickle module.

    Args:
        data (any): The data to be saved.
        filename (str): The name of the file to save to.
    """
    print(f"Saving data to '{filename}' using pickle...")
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data with pickle: {e}")

archive_path = r"C:\Users\liamk\Desktop\Random Projects\aclImdb_v1.tar.gz"

data_dir = r"C:\Users\liamk\Desktop\Random Projects\NLTK_data"
os.makedirs(data_dir, exist_ok=True)
nltk.data.path.append(data_dir)

try:
    print("Downloading 'punkt' and 'stopwords' to the specified directory...")
    # The 'download_dir' argument forces the download to the path we set
    nltk.download('punkt', download_dir=data_dir)
    nltk.download('stopwords', download_dir=data_dir)
    nltk.download('punkt_tab', download_dir=data_dir)
    print("Download complete.")
except Exception as e:
    print(f"An error occurred during download: {e}")
    print("Please check your internet connection and directory permissions.")

print(f"Starting to process reviews from {archive_path}...")
processed_reviews = process_reviews_from_archive(archive_path)
print("Finished processing.")

# Access the data using the dictionary keys
processed_corpus_train = processed_reviews['all_processed_reviews_train']
processed_corpus_test = processed_reviews['all_processed_reviews_test']
unsup_processed_corpus = processed_reviews['unsup_processed_reviews']


print(f"Supervised training corpus size: {len(processed_corpus_train)}")
print(f"Supervised testing corpus size: {len(processed_corpus_test)}")
print(f"Unsupervised training corpus size: {len(unsup_processed_corpus)}")



#SAVE PROCESSED DATA
token_data_path_train = "processed_reviews_train.pkl"
token_data_path_test = "processed_reviews_test.pkl"
untokened_data_path = "processed_unsup_reviews.pkl"
save_with_pickle(processed_corpus_train, token_data_path_train)
save_with_pickle(processed_corpus_test, token_data_path_test)
save_with_pickle(unsup_processed_corpus, untokened_data_path)