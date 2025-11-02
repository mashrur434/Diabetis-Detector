from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset


# Load model
with open('BernouliModel.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    Pregnancies = request.form.get('Pregnancies')
    Glucose= float(request.form.get('Glucose'))
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = request.form.get('Age')
    

    # Check for empty inputs
    if not all([Pregnancies, Glucose, BloodPressure,SkinThickness,Insulin , BMI,DiabetesPedigreeFunction,Age]):
        return "Error: One or more fields are empty."

    # Convert numeric fields safely
   

    # Create DataFrame
    features = pd.DataFrame([[Pregnancies, Glucose, BloodPressure,SkinThickness,Insulin , BMI,DiabetesPedigreeFunction,Age]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

    # Make prediction
    prediction = model.predict(features )[0]

    # Return the result
    result = "Yes (Diabetis  patient)" if prediction == 1 else "No (No ,Diabetis detected)"

    return render_template('index.html', prediction=result)

    

if __name__ == "__main__":
    app.run(debug=True, port=5004)
