"""
Flask chatbot application for predicting diseases based on user-provided symptoms.
The application reads symptoms from a file, corrects any misspellings or discrepancies,
and provides the most likely disease along with a description and precautions.
It also ensures at least 4 symptoms are entered before proceeding with predictions.
"""

import re
import difflib
import threading
import webbrowser
from flask import Flask, request, jsonify, render_template
from model import predict_disease


app = Flask(__name__)


# Load symptoms from file
def load_symptoms(file_path: str):
    """
    Read symptoms from a file.

    :param file_path: Path to the symptoms text file.

    :return:
        list: List of symptoms.
        """
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


def correct_symptoms(input_symptoms: list):
    """
    Correct user input symptoms by finding the closest matches from the symptom list.

    :param input_symptoms: List of user input symptoms.

    :return:
        tuple: containing:
            - corrected (list): Corrected symptoms based on the closest match.
            - incorrect (list): Symptoms that could not be matched.
    """
    corrected = []
    incorrect = []
    for symptom in input_symptoms:
        matched_symptom = difflib.get_close_matches(symptom, symptom_list, n=1)
        if matched_symptom:
            corrected.append(matched_symptom[0].title())  # Capitalize the symptom
        else:
            incorrect.append(symptom)
    return corrected, incorrect


def handle_symptom_input(symptom_input: str):
    """
    Process the user-provided symptom input, correcting symptoms and formatting them for prediction.

    :param symptom_input: Raw string of symptoms provided by the user.

    :return:
        str or dict: formatted string of corrected symptoms or dictionary
            with a response message if there are issues (e.g., not enough symptoms).
    """
    input_symptoms = re.split(r"[,]", symptom_input)
    corrected_symptoms, incorrect_symptoms = correct_symptoms(input_symptoms)

    if len(corrected_symptoms) + len(incorrect_symptoms) < 4:
        return jsonify({"response": "Please enter at least 4 symptoms!"})

    if incorrect_symptoms:
        incorrect_str = " and ".join([f'"{symptom}"' for symptom in incorrect_symptoms])
        response_msg = f"The symptoms {incorrect_str} are not in our database. Please try again!"
        return jsonify({"response": response_msg})

    corrected_str = ",".join(corrected_symptoms).replace(", ", ",")
    return corrected_str


def format_precaution(precaution_string: str):
    """
    Format the precaution string into a list and join them with line breaks.

    :param precaution_string: Contains precautions separated by commas.

    :return:
        str: Precautions separated by line breaks for display.
    """
    # precautions = precaution_string.split(", ")
    return "\n".join(precaution_string)


@app.route("/")
def index():
    """
    Route for the homepage of the application.

    :return: Renders HTML page
    """
    return render_template("index.html")


@app.route("/chat_response", methods=["POST"])
def chat_response():
    """
    Handle the response for the chatbot when the user submits symptoms.
    Processes the input symptoms, corrects any errors, predicts the disease,
    and provides a description and precautions.

    :return:
        Response: JSON response containing the disease, description, precautions,
                  or an error message if the input is invalid.
    """
    if request.method == "POST":
        try:
            symptom_input = request.form.get("data")
            formatted_symptoms = handle_symptom_input(symptom_input)

            # Check if a response is already prepared (e.g., not enough symptoms)
            if isinstance(formatted_symptoms, dict):
                return formatted_symptoms

            # Predict the disease
            prediction_result = predict_disease(formatted_symptoms)

            # Format precaution information
            formatted_precaution = format_precaution(prediction_result["Precaution"])

            doctor_message = ("It is essential to consult healthcare professionals for "
                              "accurate diagnosis and treatment!")

            return jsonify(
                {
                    "disease": prediction_result["Disease"],
                    "description": prediction_result["Description"],
                    "precaution": formatted_precaution,
                    "doctor_msg": doctor_message,
                }
            )
        except Exception as err:
            print(f"Error: {err}")
            return jsonify({"disease": "Incorrect prompt! Please try again"})
    else:
        return jsonify({})


def open_browser():
    """Automatically open the web browser when the Flask app starts."""
    webbrowser.open_new("http://127.0.0.1:5002/")


if __name__ == "__main__":
    threading.Timer(0.25, open_browser).start()
    # Load symptoms into a list
    symptom_list = load_symptoms("symptoms.txt")
    app.run(host="0.0.0.0", port=5002)
