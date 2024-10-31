import warnings
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Suppress specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


# Load the datasets
def load_data():
    """
    Load training data, symptom descriptions, and precautions from CSV files.

    Returns:
        tuple: Tuple containing the training data, symptom description, and precautions dataframes.
    """
    training_data = pd.read_csv("Training.csv").dropna(axis=1)
    symptom_description = pd.read_csv("symptom_Description.csv")
    symptom_precaution = pd.read_csv("symptom_precaution.csv")
    return training_data, symptom_description, symptom_precaution


# Encode disease labels and split the data
def prepare_data(training_data):
    """
    Encode the prognosis labels and split data into training and testing sets.

    Args:
        training_data (pd.DataFrame): The input dataset with features and prognosis.

    Returns:
        tuple: Encoded feature matrix (X), encoded labels (y), training/testing split.
    """
    label_encoder = LabelEncoder()
    training_data["prognosis"] = label_encoder.fit_transform(training_data["prognosis"])

    X = training_data.iloc[:, :-1]
    y = training_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    return X, y, X_train, X_test, y_train, y_test, label_encoder


# Initialize the models
def initialize_models():
    """
    Initialize different classifiers for the prediction task.

    Returns:
        dict: A dictionary containing the initialized classifiers.
    """
    return {
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=18)
    }


# Train models and get predictions
def train_and_predict(models, X_train, X_test, y_train):
    """
    Train the models and predict the test set.

    Args:
        models (dict): Dictionary containing initialized models.
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Test feature matrix.
        y_train (np.ndarray): Training labels.

    Returns:
        dict: Predictions from each model on the test set.
    """
    predictions = {}
    for name, model in models.items():
        model.fit(X_train.values, y_train)
        predictions[name] = model.predict(X_test.values)
    return predictions


# Train final models on the entire dataset
def train_final_models(models, X, y):
    """
    Train the models on the full dataset (without splitting).

    Args:
        models (dict): Dictionary containing initialized models.
        X (np.ndarray): Full feature matrix.
        y (np.ndarray): Full labels.
    """
    for model in models.values():
        model.fit(X.values, y)


# Map symptoms to indices
def map_symptoms_to_indices(symptoms):
    """
    Map symptom names to their corresponding column indices.

    Args:
        features (list): List of symptom feature names.

    Returns:
        dict: A dictionary mapping symptom names to indices.
    """
    symptom_mapping = {}
    for index, feature in enumerate(symptoms):
        symptom_name = " ".join([word.capitalize() for word in feature.split("_")])
        symptom_mapping[symptom_name] = index
    return symptom_mapping


# Predict disease based on symptoms
def predict_disease(symptoms):
    """
    Predict the most likely disease based on the input symptoms.

    Args:
        symptoms (str): Comma-separated string of symptoms.

    Returns:
        dict: Dictionary containing predicted disease, description, and precautions.
    """
    models, symptom_mapping, label_encoder, symptom_description, symptom_precaution = initialize_model()
    symptom_list = [symptom.replace('_', ' ') for symptom in symptoms.split(",")] # can add if-else
    input_data = [0] * len(symptom_mapping)

    # Mark the input symptoms in the feature vector
    for symptom in symptom_list:
        if symptom in symptom_mapping:
            input_data[symptom_mapping[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)

    # Get predictions from each model
    rf_prediction = label_encoder.inverse_transform(models["Random Forest"].predict(input_data))[0]
    nb_prediction = label_encoder.inverse_transform(models["Naive Bayes"].predict(input_data))[0]
    svm_prediction = label_encoder.inverse_transform(models["SVM"].predict(input_data))[0]

    # Aggregate the predictions
    predictions = np.array([rf_prediction, nb_prediction, svm_prediction])
    unique_labels, numeric_predictions = np.unique(predictions, return_inverse=True)

    # Determine the final prediction using mode
    try:
        mode_result = mode(numeric_predictions)
        final_prediction = unique_labels[mode_result.mode]
    except Exception as err:
        print(f"Prediction error: {err}")
        final_prediction = "Unknown"

    # Retrieve the disease description and precautions
    description = symptom_description.loc[symptom_description['Disease'] == final_prediction, 'description'].iloc[0]
    precautions = symptom_precaution.loc[symptom_precaution['Disease'] == final_prediction, ['1', '2', '3', '4']].iloc[
        0].to_list()

    return {
        "Disease": final_prediction,
        "Description": description,
        "Precaution": precautions
    }


# Main function to load data, train models, and make predictions
def initialize_model():
    """
    Initialize and train models, then return relevant objects for prediction.

    Returns:
        tuple: Contains the trained models, symptom mapping, and other required data for prediction.
    """
    # Load data
    training_data, symptom_description, symptom_precaution = load_data()

    # Prepare data
    X, y, X_train, X_test, y_train, y_test, label_encoder = prepare_data(training_data)

    # Initialize and train models on the full dataset
    models = initialize_models()
    train_and_predict(models, X_train, X_test, y_train)
    train_final_models(models, X, y)

    # Map symptoms to indices
    symptom_mapping = map_symptoms_to_indices(X.columns.values)

    return models, symptom_mapping, label_encoder, symptom_description, symptom_precaution

# # Main function to load data, train models, and make predictions
# if __name__ == "__main__":
#     # Load data
#     training_data, symptom_description, symptom_precaution = load_data()
#
#     # Prepare data
#     X, y, X_train, X_test, y_train, y_test, label_encoder = prepare_data(training_data)
#
#     # Initialize and train models
#     models = initialize_models()
#     train_and_predict(models, X_train, X_test, y_train)
#     train_final_models(models, X, y)
#
#     # Map symptoms to indices
#     symptom_mapping = map_symptoms_to_indices(X.columns.values)
#
#     # Example usage of prediction
#     example_symptoms = "Cough,Fever"
#     result = predict_disease(example_symptoms, symptom_mapping, models, label_encoder, symptom_description,
#                              symptom_precaution)
#     print(result)
