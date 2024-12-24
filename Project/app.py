from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the models (adjust paths to match your saved models)
models = {
    "SVM": load("../Model/SVM_model.pkl"),
    "Naive Bayes": load("../Model/Naive Bayes_model.pkl")
}

# Define the home page route
@app.route("/")
def home():
    return render_template("index.html", models=models.keys())

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs from the form
        inputs = [
            float(request.form.get("person_income")),
            float(request.form.get("person_home_ownership")),
            float(request.form.get("person_emp_length")),
            float(request.form.get("loan_grade")),
            float(request.form.get("loan_amnt")),
            float(request.form.get("loan_percent_income")),
            float(request.form.get("cb_person_default_on_file"))
        ]
        selected_model_name = request.form.get("model")

        # Get the selected model
        model = models[selected_model_name]

        # Make a prediction
        prediction = model.predict([inputs])[0]
        probability = model.predict_proba([inputs])[0][1] if hasattr(model, "predict_proba") else None

        # Return the result
        return render_template(
            "result.html",
            prediction=prediction,
            probability=probability,
            model_name=selected_model_name
        )
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
