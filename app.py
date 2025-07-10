from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    final_input = np.array([features])
    
    prediction = model.predict(final_input)
    output = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'
    
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
