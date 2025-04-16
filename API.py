from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import json
import pandas as pd

# Load the CSV file
df = pd.read_csv("C:/Users/Prahalad M/Downloads/G2 software - CRM Category Product Overviews.csv")

# Function to extract feature names from 'Features' column
def extract_feature_names(features_str):
    try:
        features_data = json.loads(features_str)
        all_feature_names = []
        for category_data in features_data:
            if 'features' in category_data and isinstance(category_data['features'], list):
                for feature in category_data['features']:
                    if 'name' in feature:
                        all_feature_names.append(f'"{feature["name"]}"')
        return all_feature_names
    except (json.JSONDecodeError, TypeError) as e:
        return []

# Function to extract text from 'pros_list' column
def extract_pros_text(pros_str):
    try:
        pros_data = json.loads(pros_str)
        pros_text = []
        for item in pros_data:
            if isinstance(item, dict) and 'text' in item:
                # Enclose the text within double quotes
                pros_text.append(f'"{item["text"]}"')
        return pros_text
    except (json.JSONDecodeError, TypeError) as e:
        return []

# Create a copy of the DataFrame
df1 = df.copy()

# Replace NaN with empty JSON array in 'Features' column
df1['Features'] = df1['Features'].fillna('[]')
# Replace NaN in rating with 0
df['rating'] = df['rating'].fillna(0)

# Apply the function to extract feature names
df1['feature_names'] = df1['Features'].apply(extract_feature_names)

# Replace NaN with empty JSON array in 'pros_list' column
df1['pros_list'] = df1['pros_list'].fillna('[]')

# Apply the function to extract pros text
df1['pros_text'] = df1['pros_list'].apply(extract_pros_text)

# Create a new column 'combined_text'
df1['all_features'] = df1.apply(lambda row: row['pros_text'] if not row['feature_names'] else row['feature_names'], axis=1)

# Create a new DataFrame with selected columns
new_df = df1[['product_name', 'seller','main_category', 'categories', 'rating', 'all_features']].copy()
new_df = new_df.rename(columns={'product_name': 'software_name', 'seller': 'vendor'})

def get_relevant_vendors(query_keywords, df, threshold=0.5):
    """
    Extracts relevant vendors and their information based on query keywords,
    feature similarity, and sorts by similarity score.
    """
    # 1. Vendor Filtering (Case-Insensitive)
    query_keywords_lower = {keyword.lower() for keyword in query_keywords}
    relevant_vendors = df[df['all_features'].apply(lambda features: any(keyword in ' '.join(features).lower() for keyword in query_keywords_lower))].copy()

    # 2. Feature Similarity
    if relevant_vendors.empty:
        return pd.DataFrame(columns=['vendor', 'software_name', 'main_category','categories', 'rating','all_features', 'similarity_score'])

    relevant_vendors = relevant_vendors.copy()
    relevant_vendors.loc[:, 'all_features_str'] = relevant_vendors['all_features'].apply(lambda x: ' '.join(x))

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(relevant_vendors['all_features_str'])

    # Calculate cosine similarity with ALL vendors
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Calculate average similarity for each vendor
    average_similarity = np.mean(similarity_matrix, axis=1)

    # Add average similarity scores to DataFrame
    relevant_vendors['similarity_score'] = average_similarity

    # 3. Threshold-Based Filtering and Sorting
    filtered_vendors = relevant_vendors[relevant_vendors['similarity_score'] >= threshold]
    filtered_vendors = filtered_vendors.sort_values(by=['similarity_score'], ascending=False)

    # Return DataFrame with desired columns, including similarity_score
    return filtered_vendors[['vendor', 'software_name', 'main_category','categories', 'rating','all_features', 'similarity_score']]

def rank_vendors(query_keywords, df, threshold=0.5, rating_weight=0.2):
    """
    Ranks vendors based on average feature similarity (weighted by individual feature match),
    overall vendor rating, and ensures search-selected vendors are included.

    Args:
        query_keywords (list): List of user-provided keywords.
        df (pd.DataFrame): DataFrame containing vendor information.
        threshold (float): Similarity threshold for initial filtering.
        rating_weight (float): Weight given to vendor rating in ranking.

    Returns:
        pd.DataFrame: Ranked vendors DataFrame with 'rank' column.
    
    """
  
    # 1. Get relevant vendors (using existing get_relevant_vendors function)
    relevant_vendors = get_relevant_vendors(query_keywords, df, threshold)

    # 2. Calculate weighted average feature similarity
    query_keywords_lower = {keyword.lower() for keyword in query_keywords}
    relevant_vendors['weighted_similarity'] = relevant_vendors['all_features'].apply(
        lambda features: sum(1 for keyword in query_keywords_lower if keyword in ' '.join(features).lower()) / len(query_keywords_lower)
    )
    relevant_vendors['weighted_similarity'] = relevant_vendors['weighted_similarity'] * (1 - rating_weight)

    # 3. Incorporate vendor rating (if available)
    if 'rating' in relevant_vendors.columns:
        relevant_vendors['rating_score'] = relevant_vendors['rating'] * rating_weight
        relevant_vendors['final_score'] = relevant_vendors['weighted_similarity'] + relevant_vendors['rating_score']
    else:
        relevant_vendors['final_score'] = relevant_vendors['weighted_similarity']

    # 4. Ensure search-selected vendors are included
    search_selected_vendors = relevant_vendors[relevant_vendors['similarity_score'] >= threshold]
    ranked_vendors = relevant_vendors.sort_values(by=['final_score'], ascending=False)

    # 5. Add rank column
    ranked_vendors['rank'] = ranked_vendors['final_score'].rank(ascending=False, method='first')
    
    if not any(query_keywords):  # Check for empty keywords
        # Sort by rating only, ignoring other factors
        ranked_vendors = df.sort_values(by=['rating'], ascending=False)  
        ranked_vendors['rank'] = ranked_vendors['rating'].rank(ascending=False, method='first')
        
    return ranked_vendors


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        category = request.form.get('category')
        capabilities = request.form.get('capabilities').split(',')  # Split capabilities by comma

        # 1. Filter by category (if provided)
        if category:
            filtered_df = new_df[new_df['categories'].str.contains(category, case=False)]
        else:
            filtered_df = new_df

        # 2. Rank vendors based on capabilities
        ranked_vendors_df = rank_vendors(capabilities, filtered_df, threshold=0.6, rating_weight=0.2)
        
        # 3. Return top 10 vendors as JSON
        top_10_vendors = ranked_vendors_df.head(10)[['vendor']]

        csv_data = top_10_vendors.to_csv(index=False)  # index=False to exclude row numbers
        return csv_data, 200, {'Content-Type': 'text/csv'}
        # For GET requests (or other methods), you can return a simple message
        return "Please send a POST request with category and capabilities."

if __name__ == '__main__':
    app.run(debug=True)