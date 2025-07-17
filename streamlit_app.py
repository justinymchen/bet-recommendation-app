import streamlit as st
import pandas as pd
import numpy as np
import re
from random import choices, sample

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

# --- Limit to 8 sports ---
allowed_sports = [
    "Basketball",
    "Football",
    "Tennis",
    "Golf",
    "Baseball",
    "American Football",
    "eSports",
]
sports = [s for s in allowed_sports if s in user_bets_df['sport'].unique()]

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
    sport_probs = {sport: sport_weights[sport] / total_weight for sport in sports}
    sport_counts = {sport: int(round(sport_probs[sport] * N_RANDOM_BETS)) for sport in sports}
    diff = N_RANDOM_BETS - sum(sport_counts.values())
    if diff != 0:
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
    np.random.shuffle(random_bets)
    st.session_state['random_bets'] = random_bets

if st.session_state['random_bets']:
    st.subheader(f"Randomized User Bet History ({N_RANDOM_BETS} bets):")
    st.dataframe(pd.DataFrame(st.session_state['random_bets']))
    st.write("These bet types will be used for recommendations.")
    bet_type_df = pd.DataFrame(st.session_state['random_bets'])
    all_bet_types = user_bets_df['bet_type'].unique()
    top_bet_types = bet_type_df['bet_type'].value_counts().head(10).index.tolist()
    # --- 10 Recommendations ---
    if st.button("Show 10 Recommendations"):
        st.subheader("Top 10 Personalized Betting Markets for You:")
        for bet_type in top_bet_types:
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
    # --- 5 Explore ---
    if st.button("Show 5 Explore Bets"):
        st.subheader("Explore New or Random Betting Markets:")
        # Find bet types not in top 10, or if not enough, sample from all
        explore_candidates = [bt for bt in all_bet_types if bt not in top_bet_types]
        if len(explore_candidates) >= 5:
            explore_bet_types = sample(list(explore_candidates), 5)
        else:
            # If not enough new, fill with random from all bet types
            explore_bet_types = list(explore_candidates)
            needed = 5 - len(explore_bet_types)
            remaining = [bt for bt in all_bet_types if bt not in explore_bet_types]
            if remaining:
                explore_bet_types += list(np.random.choice(remaining, needed, replace=False))
        for bet_type in explore_bet_types:
            # Try to find an example in the random bets, else in all history
            example = next((bet for bet in st.session_state['random_bets'] if bet['bet_type'] == bet_type), None)
            if example is None:
                example = user_bets_df[user_bets_df['bet_type'] == bet_type].iloc[0]
            with st.container():
                st.markdown(f"### {bet_type}")
                st.markdown(f"**{example['sport']} {example['league']} {example['event']}** | American Odds: {decimal_to_american(example['odds'])}")
                if st.button("Add to Betslip", key=f"explore_betslip_{bet_type}_{example['event']}"):
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