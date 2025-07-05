# app.py

# Import necessary libraries
import streamlit as st                 # For creating the web app
import joblib                          # For loading the saved model and scaler
import numpy as np                     # For handling input arrays

# ---------- Load the Pre-trained Model and Scaler ----------
# Cache the model and scaler loading to avoid reloading on every interaction
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('linear_regression_model.pkl')   # Load the trained linear regression model
    scaler = joblib.load('standard_scaler.pkl')          # Load the pre-fitted StandardScaler
    return model, scaler                                 # Return both objects

# Call the loading function and store the loaded model and scaler
lr_model, scaler = load_model_and_scaler()

# ---------- Streamlit App Title and Description ----------
st.title("Sales Prediction App")           # App title shown at the top
st.write("Enter the advertisement spend below to predict the sales.")  # App description

# ---------- User Input ----------
# Create a numeric input box for TV advertisement spend
tv_spend = st.number_input(
    'TV Advertising Spend',        # Label for the input box
    min_value=0.0,                 # Minimum allowed input value
    step=0.1                       # Increment step for the input
)

# Create a numeric input box for Radio advertisement spend
radio_spend = st.number_input(
    'Radio Advertising Spend',     # Label for the input box
    min_value=0.0,                 # Minimum allowed input value
    step=0.1                       # Increment step for the input
)

# ---------- Prediction Trigger ----------
# Add a "Predict Sales" button to perform prediction when clicked
if st.button("Predict Sales"):

    # Prepare the user inputs as a NumPy 2D array because scaler expects a 2D input
    user_input = np.array([[tv_spend, radio_spend]])

    # Apply the previously fitted scaler to scale the user input (same as during training)
    scaled_input = scaler.transform(user_input)

    # Make the sales prediction using the trained model
    predicted_sales = lr_model.predict(scaled_input)

    # Display the predicted sales as a success message
    st.success(f"Predicted Sales: {predicted_sales[0]:.2f}")

# To run this app, save it as `app.py` and use the command:
# streamlit run app.py

# Get the requirements.txt file using:
# pip freeze > requirements.txt

# # Get the requirements.txt file for the app only
# import pkg_resources
# installed_packages = pkg_resources.working_set
# with open('requirements.txt', 'w') as f:
#     for package in installed_packages:
#         f.write(f"{package.project_name}=={package.version}\n")