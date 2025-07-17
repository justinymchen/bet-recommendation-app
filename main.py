import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download latest version
path = kagglehub.dataset_download("deremmy/igaming-analysis-report")
print("Path to dataset files:", path)

# Load the dataset
csv_path = f"{path}/IGaming-Analyst-Data-Exercise.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Clean numeric columns (remove commas, spaces, convert to float)
def clean_numeric(val):
    if isinstance(val, str):
        return float(val.replace(",", "").replace(" ", "").replace("-", "0"))
    return val

for col in df.columns:
    if col not in ["year", "month", "site_id", "market"]:
        df[col] = df[col].apply(clean_numeric)

# Select relevant features for sports betting recommendation
sports_features = [
    "market",
    "sports_active_players",
    "sports_turnover_eur",
    "sports_winnings_eur",
    "sports_ggr_eur",
    "sports_bonus_issued_eur",
    "sports_bonus_withdrawn_eur",
    "sports_ngr_eur"
]

sports_df = df[sports_features].copy()

# Fill NaNs with 0
sports_df = sports_df.fillna(0)

# Normalize numeric features
scaler = MinMaxScaler()
numeric_cols = sports_features[1:]
sports_df[numeric_cols] = scaler.fit_transform(sports_df[numeric_cols])

# Function to process user input and recommend top 5 markets
import re

def recommend_bets(user_input, sports_df, numeric_cols):
    # Simple keyword extraction from user input
    keywords = re.findall(r"\w+", user_input.lower())
    # Map keywords to features (very basic mapping for demo)
    feature_weights = np.zeros(len(numeric_cols))
    for i, col in enumerate(numeric_cols):
        for word in keywords:
            if word in col:
                feature_weights[i] += 1
    # If no keywords match, use all features equally
    if feature_weights.sum() == 0:
        feature_weights[:] = 1
    # Compute a user preference vector
    user_vec = feature_weights / feature_weights.sum()
    # Compute similarity (dot product) between user_vec and each market
    market_vecs = sports_df[numeric_cols].values
    similarities = market_vecs @ user_vec
    top_indices = np.argsort(similarities)[-5:][::-1]
    recommendations = sports_df.iloc[top_indices]
    return recommendations

# Example usage:
if __name__ == "__main__":
    print("\nReady for recommendations!")
    user_input = input("Describe the types of sports bets you like: ")
    recs = recommend_bets(user_input, sports_df, numeric_cols)
    print("\nTop 5 recommended markets:")
    print(recs[["market"] + numeric_cols].to_string(index=False))
