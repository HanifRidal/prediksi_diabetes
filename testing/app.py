import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='template')

# Load the pre-trained KNeighborsClassifier model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect and validate input data
            features = []
            for field in ['feature_1', 'feature_2', 'feature_3', 'feature_4']:
                value = request.form[field]
                if not value or not value.replace('.', '', 1).isdigit():
                    raise ValueError(f"Invalid input for {field}: {value}")
                features.append(float(value))

            # Prepare the data for prediction
            features_array = np.array([features])
            prediction = model.predict(features_array)

            # Convert prediction to class name
            class_name = "Class 0" if prediction[0] == 0 else "Class 1"
            return render_template('result.html', class_name=class_name)

        except ValueError as e:
            # Handle invalid input errors
            return f"Error: {e}", 400

        except Exception as e:
            # Handle unexpected errors
            return f"An unexpected error occurred: {e}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
