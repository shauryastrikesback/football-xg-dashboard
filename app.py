
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from mplsoccer import Pitch
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Football xG Dashboard",
    layout="wide"
)

st.title("⚽ Football Expected Goals (xG) Dashboard")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("xg_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Processed Shot Data CSV",
    type=["csv"]
)

# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # ---------------------------------------------------
    # ADD MISSING COLUMNS
    # ---------------------------------------------------

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # ---------------------------------------------------
    # FEATURE MATRIX
    # ---------------------------------------------------

    X = df[feature_cols]

    # ---------------------------------------------------
    # PREDICT xG
    # ---------------------------------------------------

    df['predicted_xg'] = model.predict(X)

    # ---------------------------------------------------
    # DEBUG CHECK
    # ---------------------------------------------------

    st.subheader("xG Distribution")

    st.write(df['predicted_xg'].describe())

    # ---------------------------------------------------
    # PLAYER SELECTOR
    # ---------------------------------------------------

    players = (
        df['player']
        .dropna()
        .astype(str)
        .unique()
    )

    selected_player = st.selectbox(
        "Select Player",
        sorted(players)
    )

    player_df = df[
        df['player'] == selected_player
    ]

    # ---------------------------------------------------
    # PLAYER STATS
    # ---------------------------------------------------

    st.subheader(f"{selected_player} Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Shots",
            len(player_df)
        )

    with col2:
        st.metric(
            "Total xG",
            round(
                player_df['predicted_xg'].sum(),
                2
            )
        )

    with col3:
        st.metric(
            "Average xG",
            round(
                player_df['predicted_xg'].mean(),
                3
            )
        )

    with col4:
        st.metric(
            "Goals",
            int(player_df['is_goal'].sum())
        )

    # ---------------------------------------------------
    # FINISHING ANALYSIS
    # ---------------------------------------------------

    actual_goals = player_df['is_goal'].sum()

    expected_goals = player_df['predicted_xg'].sum()

    difference = actual_goals - expected_goals

    st.subheader("Finishing Analysis")

    st.write(
        f"Actual Goals: {actual_goals}"
    )

    st.write(
        f"Expected Goals (xG): {round(expected_goals, 2)}"
    )

    st.write(
        f"Difference: {round(difference, 2)}"
    )

    # ---------------------------------------------------
    # SHOT MAP
    # ---------------------------------------------------

    st.subheader("Shot Map")

    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='white',
        line_color='black'
    )

    fig, ax = pitch.draw(figsize=(10,7))

    scatter = pitch.scatter(
        player_df['X_m'],
        player_df['Y_m'],

        s=player_df['predicted_xg'] * 1200,

        c=player_df['predicted_xg'],

        cmap='Reds',

        edgecolors='black',

        ax=ax
    )

    plt.title(
        f"{selected_player} Shot Map"
    )

    plt.colorbar(scatter)

    st.pyplot(fig)

    # ---------------------------------------------------
    # TOP PLAYERS LEADERBOARD
    # ---------------------------------------------------

    leaderboard = (
        df.groupby('player')['predicted_xg']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.subheader("Top Players by Total xG")

    st.dataframe(leaderboard)

    # ---------------------------------------------------
    # PREDICTIONS TABLE
    # ---------------------------------------------------

    st.subheader("Player Predictions")

    st.dataframe(
        player_df[
            [
                'player',
                'predicted_xg',
                'is_goal'
            ]
        ].head(20)
    )
