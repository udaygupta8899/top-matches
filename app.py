import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import numpy as np

# Function to preprocess the dataset
def preprocess_data(data):
    data_processed = data.copy()
    encoders = {}
    
    # Encode categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data[col])
        encoders[col] = le
    
    # Normalize numerical columns
    scaler = MinMaxScaler()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data_processed[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data_processed, encoders, scaler

# Improved Weighted Similarity Calculation
def calculate_weighted_similarity(row1, row2, data, weights):
    similarity = 0
    total_weight = sum(weights.values())
    
    for col in data.columns:
        weight = weights.get(col, 0)
        if weight == 0:
            continue
        
        if data[col].dtype in ['int64', 'float64']:  # Numerical column
            # Relative scaling for numerical columns
            col_range = data[col].max() - data[col].min()
            if col_range > 0:  # Avoid division by zero
                diff = abs(row1[col] - row2[col]) / col_range
            else:
                diff = 0  # identical values

            similarity += (1 - diff) * weight  # Scale difference to similarity (max 1)

        else:  # Categorical column
            # Jaccard similarity for categorical data
            sim = 1 - jaccard([row1[col]], [row2[col]])  # Jaccard similarity for categorical values
            similarity += sim * weight
    
    # Normalize similarity to ensure it falls between 0 and 100
    normalized_similarity = (similarity / total_weight) * 100
    return max(0, min(normalized_similarity, 100))  # Ensure the result is between 0 and 100

# Streamlit App
st.title("FDS MINOR PROJECT")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Preprocess data
    data_processed, encoders, scaler = preprocess_data(data)
    
    # Define column weights
    weights = {col: 1 for col in data.columns}  # Default weights
    st.sidebar.header("Adjust Column Weights")
    for col in data.columns:
        weights[col] = st.sidebar.slider(f"Weight for {col}", 0, 10, 1)

    # User input form for questionnaire
    st.subheader("Fill the questionnaire based on the columns")
    user_input = {}
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            user_input[col] = st.number_input(f"Enter value for {col}", value=1,min_value=1,max_value=10)
        else:
            user_input[col] = st.selectbox(f"Select value for {col}", options=data[col].unique())

    # Encode and scale user input to match the preprocessed data
    user_input_processed = user_input.copy()
    for col, encoder in encoders.items():
        if col in user_input_processed:
            user_input_processed[col] = encoder.transform([user_input[col]])[0]

    user_input_processed = pd.DataFrame([user_input_processed])
    user_input_processed[data.select_dtypes(include=['int64', 'float64']).columns] = scaler.transform(user_input_processed[data.select_dtypes(include=['int64', 'float64']).columns])

    # Calculate similarity for each row and get top 5 matches
    similarities = []
    for i, row in data_processed.iterrows():
        similarity = calculate_weighted_similarity(user_input_processed.iloc[0], row, data_processed, weights)
        similarities.append((i, similarity))

    # Sort similarities and get top 5 matches
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_matches = similarities[:5]

    # Display top 5 matches
    st.subheader("Top 5 Similarity Matches")
    for index, similarity in top_5_matches:
        st.write(f"Row {index} with similarity: {similarity:.2f}%")
        st.write(data.iloc[index])

    # Display the user's input details
    st.subheader("Your Input Details")
    st.write(user_input)
