import pandas as pd
import numpy as np
import streamlit as st
import pickle


imported_data = pd.read_csv('imported_data.csv')
# Load models and encoders
with open('X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('random_forest_regressor.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('Extree_clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open ('model_metrics.pkl','rb') as f:
    clf_model_metrics = pickle.load(f)
reg_metrics = {
    "Extra Tree Regressor": {'R2': 0.35346726339696477, 'MSE': 0.13012589658985954, 'MAE': 0.14695558807392867},
    "Random Forest": {'R2': 0.43500584933926856, 'MSE': 0.11371484576177718, 'MAE': 0.14151056849977137},
    "Linear Regression": {'R2': 0.03679714967891079, 'MSE': 0.19386123455167917, 'MAE': 0.213038783258982},
    "Ridge Regression": {'R2': 0.03679715745362411, 'MSE': 0.19386123298688362, 'MAE': 0.21303881348902448},
    "Lasso Regression": {'R2': -6.994911709323759e-05, 'MSE': 0.201281375890079, 'MAE': 0.22925763835586047}
}

# Create a DataFrame from the dictionary
df_regression_metrics = pd.DataFrame(reg_metrics).transpose()

# Reset the index to have model names as a column
df_regression_metrics.reset_index(inplace=True)

# Rename the columns
df_regression_metrics.columns = ['Model', 'R2 Score', 'Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)']


# Streamlit app layout
st.title('Industrial Copper Modelling')

# Create a sidebar with tabs
tab = st.sidebar.selectbox('Choose a tab', ['Model Metrics', 'Prediction'])


if tab == 'Model Metrics':
    st.header("Classification Model Evaluation Metrics")
    clf_model_metrics_df = pd.DataFrame(clf_model_metrics).transpose()
    clf_model_metrics_df.reset_index(inplace=True)
    clf_model_metrics_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    st.dataframe(clf_model_metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']))
    st.header("Regression Model Evaluation Metrics")
    reg_model_metrics_df = df_regression_metrics
    st.dataframe(reg_model_metrics_df.style.highlight_max(subset=['R2 Score'], axis=0)
                                         .highlight_min(subset=['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)'], axis=0))

elif tab == 'Prediction':
    # Input features in three columns
    with st.form('input_form'):
        input_data = {}
        first_row = imported_data.iloc[181527]
        cols = st.columns(3)  # Create 3 columns

        for i, col in enumerate(X.columns):
            col_index = i % 3  # Determine column index (0, 1, or 2)
            with cols[col_index]:
                input_data[col] = st.text_input(col, value=str(first_row[col]))

        # Submit buttons for the form
        predict_price_button = st.form_submit_button(label='Predict Selling Price')
        predict_status_button = st.form_submit_button(label='Predict Status')

    if predict_price_button or predict_status_button:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Handle empty inputs by replacing with default values
        input_df.replace('', np.nan, inplace=True)
        input_df.fillna(0, inplace=True)

        # Apply label encoding
        for column in ['status', 'item type', 'material_ref']:
            if column in input_df.columns:
                le = label_encoders[column]
                input_df[column] = input_df[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Apply log transformation (if required)
        skewed_features = ['thickness', 'country', 'width', 'selling_price']  # Replace with the actual list of skewed features
        for feature in skewed_features:
            if feature in input_df.columns:
                input_df[feature] = np.log1p(input_df[feature].astype(float))

        # Standardize features
        input_scaled = scaler.transform(input_df)

        # Make predictions and highlight the result
        if predict_price_button:
            prediction = reg_model.predict(input_scaled)
            st.markdown(f'**Predicted Selling Price: {np.expm1(prediction)[0]:.2f}**')

        if predict_status_button:
            prediction = clf_model.predict(input_scaled)
            st.markdown(f'**Predicted Status: {"WON" if prediction[0] == 7 else "LOST"}**')
