
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from mplsoccer import Pitch
import matplotlib.pyplot as plt

# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("⚽ Football xG Dashboard")

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("xg_model.pkl")

feature_cols = joblib.load("feature_cols.pkl")

# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Shot Data CSV",
    type=["csv"]
)

# -----------------------------
# MAIN APP
# -----------------------------

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------

    GOAL_X = 105
    GOAL_Y = 34

    df['X_m'] = df['X'] * 105
    df['Y_m'] = df['Y'] * 68

    df['shot_distance'] = np.sqrt(
        (GOAL_X - df['X_m'])**2 +
        (df['Y_m'] - GOAL_Y)**2
    )

    df['shot_angle'] = np.arctan2(
        np.abs(df['Y_m'] - GOAL_Y),
        (GOAL_X - df['X_m'])
    )

    df['dist2'] = df['shot_distance'] ** 2
    df['angle2'] = df['shot_angle'] ** 2

    df['angle_distance'] = (
        df['shot_distance'] *
        df['shot_angle']
    )

    # -----------------------------
    # ADD MISSING COLUMNS
    # -----------------------------

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # -----------------------------
    # PREDICT xG
    # -----------------------------

    X = df[feature_cols]

    df['predicted_xg'] = model.predict(X)

    # -----------------------------
    # PLAYER SELECTOR
    # -----------------------------

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

    # -----------------------------
    # PLAYER STATS
    # -----------------------------

    st.subheader("Player Statistics")

    col1, col2, col3 = st.columns(3)

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

    # -----------------------------
    # SHOT MAP
    # -----------------------------

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

    plt.title(f"{selected_player} Shot Map")

    plt.colorbar(scatter)

    st.pyplot(fig)

    # -----------------------------
    # DATA TABLE
    # -----------------------------

    st.subheader("Predictions")

    st.dataframe(
        player_df[
            [
                'player',
                'predicted_xg'
            ]
        ]
    )
