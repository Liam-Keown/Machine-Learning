import pickle
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import csv

def load_with_pickle(filename):
    """
    Loads a Python object from a file using the pickle module.

    Args:
        filename (str): The name of the file to load from.

    Returns:
        any: The loaded data object, or None if an error occurred.
    """
    print(f"\nLoading data from '{filename}' using pickle...")
    try:
        # 'rb' mode is for reading in binary
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"Error loading data with pickle: {e}")
        return None
    
def TF_IDF(training_data: List[Tuple[List[str], str, str]], testing_data: List[Tuple[List[str], str, str]]):
    print("\nApplying TF-IDF conversion to the datasets...")

    train_texts = [" ".join(review_tuple[0]) for review_tuple in training_data]
    test_texts = [" ".join(review_tuple[0]) for review_tuple in testing_data]

    labels_train = [review_tuple[1] for review_tuple in training_data]
    labels_test = [review_tuple[1] for review_tuple in testing_data]

    vectoriser = TfidfVectorizer(max_features=25000, ngram_range=(1,2))

    train_input = vectoriser.fit_transform(train_texts)
    test_input = vectoriser.transform(test_texts)

    print("TF-IDF conversion complete.")
    
    return train_input, labels_train, test_input, labels_test


def train_and_evaluate_model(train_features, train_labels, test_features, test_labels):
    """
    Trains a Multinomial Naive Bayes classifier and evaluates its performance.

    Args:
        train_features: The TF-IDF features for the training data.
        train_labels: The sentiment labels for the training data.
        test_features: The TF-IDF features for the testing data.
        test_labels: The sentiment labels for the testing data.
    """
    print("\nTraining the classifier model...")

    model = LinearSVC(dual="auto")

    model.fit(train_features, train_labels)

    print("Model training complete.")
    print("Evaluating the model on the test data...")

    predictions = model.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)

    print(f"\nModel Accuracy on the test set: {accuracy:.4f}")

    return predictions, test_labels, test_features, accuracy


#==============Main=================
review_path_train = "processed_reviews_train.pkl"
review_path_test = "processed_reviews_test.pkl"

nlp_data_train = load_with_pickle(review_path_train)
nlp_data_test = load_with_pickle(review_path_test)

# print(len(nlp_data_train), len(nlp_data_test)) ---- 25000 each!

test_data_size = 25000 #cant be greater than 25000

nlp_data_train = nlp_data_train + nlp_data_test[:len(nlp_data_test)-test_data_size]
nlp_data_test = nlp_data_test[len(nlp_data_test)-test_data_size:]

nlp_data_train = shuffle(nlp_data_train)
nlp_data_test = shuffle(nlp_data_test)

if nlp_data_train and nlp_data_test:
    train_features, train_labels, test_features, test_labels = TF_IDF(nlp_data_train, nlp_data_test)

    test_predictions, test_labels, test_features, score = train_and_evaluate_model(train_features, train_labels, test_features, test_labels)


    file_path = 'nlp_output_results.csv'

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"This model achieved {score*100}% of correct results."])
        writer.writerow(["Prediction","Answer"])
        inspection_size = len(test_predictions)
        for index,prediction in enumerate(test_predictions[:inspection_size]):
            writer.writerow([prediction, test_labels[index]])

    print(f"Predictions and targets have been saved to {file_path}")



