from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/api", methods=['POST'])
def PCOS_predict():
    # Parse the JSON request data
    data = request.get_json()

    # Extract the values from the JSON data
    input_features = [
        data["BMI"], data["Cycle(R/I)"], data["Cycle length(days)"], data["Pregnant(Y/N)"],
        data["No. of aborptions"], data["Hip(inch)"], data["Waist(inch)"], data["Waist:Hip Ratio"],
        data["Weight gain(Y/N)"], data["hair growth(Y/N)"], data["Skin darkening (Y/N)"],
        data["Pimples(Y/N)"], data["Fast food (Y/N)"], data["Follicle No. (L)"], data["Follicle No. (R)"]
    ]
    
    # Convert input_features to the format expected by the model
    input_features = np.array(input_features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)

    # Prepare the response dictionary
    response = {"PCOS_Y/N": int(prediction[0])}

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
