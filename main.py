# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
from huggingface_hub import login
from tenacity import retry, stop_after_attempt, wait_fixed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Install necessary packages
# !pip install transformers torch scikit-learn pandas nltk sentencepiece tenacity

# Log in to Hugging Face
access_token = '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'
login(token=access_token, add_to_git_credential=True)

# Define model identifier
model_identifier = 'gpt2'

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def load_model_and_tokenizer(model_identifier, access_token):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_identifier, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_identifier, use_auth_token=access_token)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        raise

# Load the tokenizer and model with retry logic
tokenizer, model = load_model_and_tokenizer(model_identifier, access_token)

# Load the crime data from a CSV file
# Assume the dataset contains columns: ['description', 'latitude', 'longitude', 'crime_type', 'date']
crime_data = pd.read_csv('crime_data.csv')

# Basic text cleaning: lowercase, remove special characters
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text

# Check if 'description' column exists
if 'description' in crime_data.columns:
    # Clean the crime descriptions
    crime_data['clean_description'] = crime_data['description'].apply(clean_text)

    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    crime_data['tokens'] = crime_data['clean_description'].apply(
        lambda x: [word for word in word_tokenize(x) if word not in stop_words])

    # Display cleaned text data
    print(crime_data[['description', 'clean_description']].head())
else:
    print("Column 'description' does not exist in crime_data. Available columns are:", crime_data.columns)

# Example crime_data DataFrame
crime_data = pd.DataFrame({
    'clean_description': ["description1", "description2"],
    'latitude': [34.0522, 36.1699],
    'longitude': [-118.2437, -115.1398],
    'crime_type': ['theft', 'assault']
})

# Tokenize the cleaned crime descriptions
inputs = tokenizer(crime_data['clean_description'].tolist(), return_tensors='pt', padding=True, truncation=True)

# Generate embeddings for the crime descriptions
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.logits

# Convert embeddings to a 2D array
embeddings = embeddings.mean(dim=1).cpu().numpy()

# Combine embeddings with geographical data (latitude, longitude)
features = np.hstack([embeddings, crime_data[['latitude', 'longitude']].values])

# The target variable could be future crime occurrences, crime types, etc.
# For simplicity, we'll assume we're predicting the type of crime
target = crime_data['crime_type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Prediction Accuracy: {accuracy}")

# New sample data for prediction (new crime descriptions and locations)
new_data = {
    'description': ["Suspicious activity near a shopping mall", "Attempted theft in residential area"],
    'latitude': [41.881832, 41.878113],
    'longitude': [-87.623177, -87.629799]
}

# Preprocess and tokenize the new data
new_data_df = pd.DataFrame(new_data)
new_data_df['clean_description'] = new_data_df['description'].apply(clean_text)
new_inputs = tokenizer(new_data_df['clean_description'].tolist(), return_tensors='pt', padding=True, truncation=True)

# Generate embeddings for the new data
with torch.no_grad():
    new_outputs = model(**new_inputs)
    new_embeddings = new_outputs.logits

# Convert new embeddings to a 2D array
new_embeddings = new_embeddings.mean(dim=1).cpu().numpy()

# Combine new embeddings with geographical data
new_features = np.hstack([new_embeddings, new_data_df[['latitude', 'longitude']].values])

# Predict crime types or hotspot potential for new data
new_predictions = clf.predict(new_features)
print("Predicted Crime Types:", new_predictions)