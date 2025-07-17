import streamlit as st
import pandas as pd
import numpy as np
import re
from random import choices

# Helper to convert decimal odds to American odds
def decimal_to_american(decimal_odds):
    try:
        decimal_odds = float(decimal_odds)
        if decimal_odds >= 2:
            return f"+{int((decimal_odds - 1) * 100)}"
        else:
            return f"-{int(100 / (decimal_odds - 1))}"
    except:
        return "-"

# Load the user's bet history dataset
@st.cache_data
def load_bets():
    df = pd.read_csv("user_bet_history.csv")
    return df

user_bets_df = load_bets()

st.title("Personalized Betting Market Recommendation Engine")

# --- Sport Distribution Sliders ---
sports = sorted(user_bets_df['sport'].unique())
st.sidebar.header("Randomized Bet Distribution by Sport")
sport_weights = {}
total_weight = 0
for sport in sports:
    weight = st.sidebar.slider(f"{sport}", min_value=0, max_value=100, value=20)
    sport_weights[sport] = weight
    total_weight += weight
if total_weight == 0:
    st.warning("Please set at least one sport weight above 0.")

# --- Betslip State ---
if 'betslip' not in st.session_state:
    st.session_state['betslip'] = []

# --- Randomize Section ---
if 'random_bets' not in st.session_state:
    st.session_state['random_bets'] = []

N_RANDOM_BETS = 500

if st.button(f"Randomize {N_RANDOM_BETS} Bets") and total_weight > 0:
    # Calculate probabilities for each sport
    sport_probs = {sport: sport_weights[sport] / total_weight for sport in sports}
    # Determine how many bets to sample for each sport
    sport_counts = {sport: int(round(sport_probs[sport] * N_RANDOM_BETS)) for sport in sports}
    # Adjust to ensure total is exactly N_RANDOM_BETS
    diff = N_RANDOM_BETS - sum(sport_counts.values())
    if diff != 0:
        # Adjust the largest sport by the difference
        max_sport = max(sport_counts, key=sport_counts.get)
        sport_counts[max_sport] += diff
    random_bets = []
    for sport, count in sport_counts.items():
        pool = user_bets_df[user_bets_df['sport'] == sport]
        if pool.empty:
            continue
        if count > len(pool):
            st.warning(f"Not enough unique bets for {sport}. Some bets will be repeated.")
        sampled = pool.sample(count, replace=True).to_dict('records')
        random_bets.extend(sampled)
    # Shuffle the combined list
    np.random.shuffle(random_bets)
    st.session_state['random_bets'] = random_bets

if st.session_state['random_bets']:
    st.subheader(f"Randomized User Bet History ({N_RANDOM_BETS} bets):")
    st.dataframe(pd.DataFrame(st.session_state['random_bets']))
    random_bet_types = [bet['bet_type'] for bet in st.session_state['random_bets']]
    user_bets = "\n".join(random_bet_types)
    st.write("These bet types will be used for recommendations.")
    if st.button("Get Recommendations"):
        # Only consider bet types present in the random bets
        bet_type_df = pd.DataFrame(st.session_state['random_bets'])
        bet_types_in_random = bet_type_df['bet_type'].unique()
        # Score each bet_type by number of keyword matches (using the random bets as input)
        keywords = re.findall(r"\w+", " ".join(bet_types_in_random).lower())
        bet_type_scores = {}
        for bet_type in bet_types_in_random:
            score = sum(1 for word in keywords if word in str(bet_type).lower())
            bet_type_scores[bet_type] = score
        # Engagement: most frequent bet types in the random bets, weighted by average wager amount
        engagement_scores = bet_type_df.groupby('bet_type')['amount'].mean() * bet_type_df['bet_type'].value_counts()
        engagement_scores = engagement_scores.loc[bet_types_in_random].sort_values(ascending=False)
        # Blend: top 10 by similarity, fill with most engaged if needed
        top_similar = sorted(bet_type_scores, key=bet_type_scores.get, reverse=True)[:10] if bet_type_scores else []
        if len(top_similar) < 10:
            top_engaged = [bt for bt in engagement_scores.index if bt not in top_similar][:10-len(top_similar)]
            top_bet_types = list(top_similar) + list(top_engaged)
        else:
            top_bet_types = list(top_similar)
        if not top_bet_types:
            top_bet_types = list(engagement_scores.head(10).index)
            st.info("No close matches found. Showing your most engaged betting markets.")
        st.subheader("Top 10 Predicted Betting Markets for You:")
        for bet_type in top_bet_types:
            # Find an example from the random bets
            example = next((bet for bet in st.session_state['random_bets'] if bet['bet_type'] == bet_type), None)
            if example is not None:
                with st.container():
                    st.markdown(f"### {bet_type}")
                    st.markdown(f"**{example['sport']} {example['league']} {example['event']}** | American Odds: {decimal_to_american(example['odds'])}")
                    if st.button("Add to Betslip", key=f"betslip_{bet_type}_{example['event']}"):
                        st.session_state['betslip'].append({
                            'bet_type': bet_type,
                            'sport': example['sport'],
                            'league': example['league'],
                            'event': example['event'],
                            'odds': decimal_to_american(example['odds'])
                        })
        # Show betslip
        if st.session_state['betslip']:
            st.subheader("Your Betslip")
            betslip_df = pd.DataFrame(st.session_state['betslip'])
            st.dataframe(betslip_df)
else:
    st.info(f"Use the sliders in the sidebar to set your sport preferences, then click 'Randomize {N_RANDOM_BETS} Bets'.") 