import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# --- 1. Data Cleaning and Preprocessing Functions ---

def convert_price_to_rupees(price):
    """Converts price strings (e.g., '3.05 Cr', '68.75 L') to a uniform value in Indian Rupees."""
    if isinstance(price, str):
        price = price.replace(',', '').strip()
        if 'Cr' in price:
            # Crore to Rupee: Multiply by 10,000,000
            return float(price.replace('Cr', '').strip()) * 10_000_000
        elif 'L' in price:
            # Lakh to Rupee: Multiply by 100,000
            return float(price.replace('L', '').strip()) * 100_000
    try:
        return float(price)
    except:
        return np.nan

def clean_data(df):
    """Performs the necessary data cleaning and feature engineering."""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # 1. Clean the target variable 'price'
    df['price_in_rupees'] = df['price'].apply(convert_price_to_rupees)

    # Remove rows where price conversion failed or is missing
    df.dropna(subset=['price_in_rupees'], inplace=True)

    # 2. Clean the 'BHK' (Bedrooms) column
    # Extract the first digit (the number of bedrooms)
    df['BHK_count'] = df['BHK'].str.extract(r'(\d+)').astype(float)
    df.dropna(subset=['BHK_count'], inplace=True)
    df['BHK_count'] = df['BHK_count'].astype(int)

    # 3. Handle 'Total sqft'
    # 'Total sqft' is already clean (int64)

    # 4. Filter out extreme outliers (e.g., extremely high prices that can skew the model)
    # A simple quantile-based filter (e.g., keep prices within 99th percentile)
    price_upper_bound = df['price_in_rupees'].quantile(0.99)
    df = df[df['price_in_rupees'] < price_upper_bound]

    # Select final features and target
    X = df[['BHK_count', 'Total sqft', 'Location']]
    y = df['price_in_rupees']

    return X, y, df

# --- 2. Streamlit App Layout and Logic ---

# Set a wide page configuration
st.set_page_config(layout="wide")

# App Title and Description
st.title("🏡 Mumbai Real Estate Price Predictor")
st.markdown("""
A machine learning application to estimate house prices in Mumbai based on property features.
*Built with Python, Streamlit, and Scikit-learn.*
---
""")

@st.cache_data
def load_data(file_path):
    """Caches data loading to improve performance."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

@st.cache_resource
def train_model(X, y):
    """Trains and caches the machine learning pipeline."""
    
    # Define the preprocessing steps
    # We will One-Hot Encode the 'Location' column
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Location'])
        ],
        remainder='passthrough' # Keep 'BHK_count' and 'Total sqft' as they are
    )

    # Create a pipeline with preprocessing and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5))
    ])

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model_pipeline.fit(X_train, y_train)

    return model_pipeline, X # Return X to get the unique locations for the sidebar

# --- Main Execution ---
df_raw = load_data("house_price_mumbai.csv")
X, y, df_clean = clean_data(df_raw)
model, X_features = train_model(X, y)

# --- 3. Sidebar for User Input ---
st.sidebar.header("🔍 Property Features Input")

# Get unique locations for the dropdown
unique_locations = sorted(X_features['Location'].unique())

# Input fields
bhk_input = st.sidebar.slider("Number of Bedrooms (BHK)",
                              min_value=1,
                              max_value=10,
                              value=2,
                              step=1)

sqft_input = st.sidebar.number_input("Total Area (Sq. Ft.)",
                                      min_value=100,
                                      max_value=10000,
                                      value=800,
                                      step=50)

location_input = st.sidebar.selectbox("Location (Suburb)", unique_locations)

# Create input DataFrame for prediction
input_data = pd.DataFrame([[bhk_input, sqft_input, location_input]],
                          columns=['BHK_count', 'Total sqft', 'Location'])

# Prediction Button
if st.sidebar.button("Predict Price"):
    try:
        # --- 4. Prediction Logic ---
        # Calculate real average price per sqft for this location from the dataset
        location_data = df_clean[df_clean['Location'] == location_input]
        if not location_data.empty:
            avg_price_per_sqft = (location_data['price_in_rupees'] / location_data['Total sqft']).mean()
        else:
            # Fallback to ML model if location not found in dataset
            ml_prediction = model.predict(input_data)[0]
            avg_price_per_sqft = ml_prediction / sqft_input

        # Calculate estimated price based on: Price = Square Feet × Price per Sq Ft
        estimated_price = sqft_input * avg_price_per_sqft

        # Convert prediction to Crore and Lakh for better readability
        if estimated_price >= 10_000_000:
            # Display in Crore
            price_str = f"**₹{estimated_price / 10_000_000:,.2f} Crore**"
        else:
            # Display in Lakh
            price_str = f"**₹{estimated_price / 100_000:,.2f} Lakhs**"

        st.subheader("💰 Estimated Property Price")
        st.success(f"The estimated price for this property is:")
        st.markdown(f"## {price_str}")
        
        # Display price per sqft and location
        st.info(f"**Price per Sq. Ft:** ₹{avg_price_per_sqft:,.0f}")
        st.info(f"**Location:** {location_input}")
        
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Note: The original 'price_sqft' and 'project' columns were excluded to simplify the model,
# as 'price_sqft' is a derivative of the target and 'project' has high cardinality.