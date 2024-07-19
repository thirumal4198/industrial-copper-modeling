# Industrial Copper Modeling

## Description
This project involves predicting the selling price and status of industrial copper using machine learning models. It includes both regression and classification tasks to help manufacturers and stakeholders make informed decisions based on predictive analytics.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thirumal4198/industrial-copper-modeling.git
2. Navigate to the project directory:
cd industrial-copper-modeling

3. Create a virtual environment and activate it:
   python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

4. Install the required dependencies:
   pip install -r requirements.txt


## Usage
Running the Streamlit App
To run the Streamlit app, execute the following command:
streamlit run cu.py

## Predicting Selling Price and Status
You can use the trained models to make predictions on new data. Here is an example:
import pandas as pd
import pickle

## Load the models
with open('random_forest_regressor.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('Extree_clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

## Sample data
data = pd.read_csv('***.csv')


## Preprocess the data (scaling, encoding, etc.)

## Predict selling price
selling_price_prediction = reg_model.predict(data)
print(f"Predicted Selling Price: {selling_price_prediction[0]}")

## Predict status
status_prediction = clf_model.predict(data)
print(f"Predicted Status: {'WON' if status_prediction[0] == 7 else 'LOST'}")


## Project Structure
industrial-copper-modeling/
├── data/
│   ├── imported_data.csv
├── models/
│   ├── random_forest_regressor.pkl
│   ├── Extree_clf_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
├── notebooks/
│   ├── cu.ipynb
├── app/
│   ├── cu.py
├── requirements.txt
├── README.md



## Model Details
### Regression_Models:
Model_Metrics= 
    
    "Extra Tree Regressor": {'R2': 0.35346726339696477, 'MSE': 0.13012589658985954,'MAE': 0.14695558807392867},
    "Random Forest": {'R2': 0.43500584933926856, 'MSE': 0.11371484576177718, 'MAE': 0.14151056849977137},
    "Linear Regression": {'R2': 0.03679714967891079, 'MSE': 0.19386123455167917, 'MAE': 0.213038783258982},
    "Ridge Regression": {'R2': 0.03679715745362411, 'MSE': 0.19386123298688362, 'MAE': 0.21303881348902448},
    "Lasso Regression": {'R2': -6.994911709323759e-05, 'MSE': 0.201281375890079, 'MAE': 0.22925763835586047}



Classification Models:
model_metrics = 

    "Extra Trees Classifier": {
        "Accuracy": 0.7924,
        "Precision": 0.7880,
        "Recall": 0.7924,
        "F1-Score": 0.7900
    },
    "Logistic Regression": {
        "Accuracy": 0.6599,
        "Precision": 0.5286,
        "Recall": 0.6599,
        "F1-Score": 0.5699
    },
    "Support Vector Machine": {
        "Accuracy": 0.6822,
        "Precision": 0.6352,
        "Recall": 0.6822,
        "F1-Score": 0.6131
    },
    "Gradient Boosting Classifier": {
        "Accuracy": 0.6563,
        "Precision": 0.6264,
        "Recall": 0.6563,
        "F1-Score": 0.6314
    }

## Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.


## License
This project is licensed under the MIT License - see the LICENSE file for details.
You can copy and paste this entire block into your GitHub README.md editor.

