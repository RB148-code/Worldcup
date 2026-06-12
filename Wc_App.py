import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
from datetime import datetime, timedelta, timezone
import base64

#Page configurations
st.set_page_config(
    page_title="2026 World Cup Hub",
    page_icon= 'logo.png',
    layout="wide"
)
scaler = joblib.load('R:\\H.H.H\\FIFA  WORLD CUP 2026\\2026 data\\scaler.pkl')

st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            /* Use direct image URL or correct path */
            background-image: url('file:///R:/H.H.H/FIFA WORLD CUP 2026/images_cup.jpg') !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            background-color: #1a0000 !important; /* Fallback if image fails */
            border-right: 1px solid rgba(255, 215, 0, 0.25) !important;
            position: relative !important;
        }

        /* Dark overlay to keep text readable */
        [data-testid="stSidebar"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.75); /* Darken so text is visible */
            z-index: 0;
            pointer-events: none;
        }

        /* Bring all content above overlay */
        [data-testid="stSidebar"] > div:first-child {
            position: relative !important;
            z-index: 1 !important;
        }

        /* Text styling */
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* Selectbox styling */
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: rgba(139, 0, 0, 0.7) !important;
            border: 1px solid rgba(255, 215, 0, 0.4) !important;
            border-radius: 8px !important;
            backdrop-filter: blur(4px);
        }

        [data-testid="stSidebar"] .stSelectbox > div:hover {
            background-color: rgba(220, 20, 20, 0.8) !important;
        }
        /* Popup overlay - adjusted to avoid top bar */
        .popup-overlay {
            position: fixed;
            top: 60px; /* Leaves space for Streamlit's top bar */
            left: 0;
            width: 100%;
            height: calc(100% - 60px);
            background: rgba(0,0,0,0.7);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* Minimum compact popup */
        .popup-box {
            background: linear-gradient(135deg, #2563eb, #9333ea, #ec4899);
            border-radius: 16px;
            padding: 1.5rem;
            width: 90%;
            max-width: 380px;
            text-align: center;
            position: relative;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        /* Visible close button */
        .close-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: #ef4444;
            color: white;
            border: none;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
            z-index: 10000;
        }
        .close-btn:hover {
            background: #dc2626;
        }
        .wc-badge {
            background: linear-gradient(90deg, #3b82f6, #a855f7, #ec4899);
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 18px;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 0.8rem;
            font-size: 0.9rem;
        }
        .countdown-title {
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0 0 0.3rem 0;
            color: #ffffff;
        }
        .countdown-subtitle {
            font-size: 0.9rem;
            color: #bbf7d0;
            margin-bottom: 1.2rem;
        }
        .countdown-grid {
            display: flex;
            justify-content: center;
            gap: 0.7rem;
        }
        .countdown-box {
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
            width: 75px;
            padding: 0.8rem 0.2rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .countdown-number {
            font-size: 1.8rem;
            font-weight: bold;
            display: block;
            color: #ffffff;
        }
        .countdown-label {
            font-size: 0.75rem;
            opacity: 0.9;
            text-transform: uppercase;
            margin-top: 0.2rem;
        }
        .timezone-note {
            margin-top: 1rem;
            font-size: 0.75rem;
            color: #bfdbfe;
            opacity: 0.9;
        }
        /* Flag styling */
        .flag-card {
            text-align: center;
            color: white;
            padding: 0.4rem;
        }
        .flag-img {
            border-radius: 4px;
            border: 2px solid #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            width: 45px;
            height: auto;
        }
        .flag-name {
            font-size: 0.7rem;
            margin-top: 0.2rem;
            opacity: 0.9;
        }
        [data-testid="stButton"] {
        position: relative;
        z-index: 10001;
        }
        </style>
    """, unsafe_allow_html=True)

#Dat and model loading
@st.cache_data  # Caches data so it loads fast
def load_data():
    df = pd.read_csv("R:\\H.H.H\\FIFA  WORLD CUP 2026\\2026 data\\players_info\\transfermarkt_players_clean.csv")
    return df

@st.cache_resource  # Caches models so they don't reload every time
def load_models():
    home_model = joblib.load("R:\\H.H.H\\FIFA  WORLD CUP 2026\\2026 data\\home_score_model.pkl")
    away_model = joblib.load("R:\\H.H.H\\FIFA  WORLD CUP 2026\\2026 data\\away_score_model.pkl")
    match_data = pd.read_csv("R:\\H.H.H\\FIFA  WORLD CUP 2026\\2026 data\\Clean Updated data of 2026.csv")
    return home_model, away_model, match_data

# Load everything
players_df = load_data()
home_model, away_model, match_data = load_models()
standings = pd.read_csv('R:\\H.H.H\\FIFA  WORLD CUP 2026\\2026 data\\Fifa cup standings.csv')

#clustering
@st.cache_resource
def cluster_players(df):
    style_features = [
        "goals_per90", "assists_per90", "pass_accuracy", "progressive_passes",
        "dribbles_per90", "duels_won_pct", "tackles_per90", "interceptions_per90",
        "aerial_duels_won_pct"
    ]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[style_features])
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
    df["style_cluster"] = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    df[["pca1", "pca2"]] = pca.fit_transform(X_scaled)
    
    style_names = {
        0: "⚽ Finishing Forward",
        1: "🎯 Creative Playmaker",
        2: "🛡️ Defensive Anchor",
        3: "💨 Dynamic Winger",
        4: "🧱 Ball-Playing Defender",
        5: "🔄 Box-to-Box Midfielder"
    }
    df["style_label"] = df["style_cluster"].map(style_names)
    return df

players_df = cluster_players(players_df)

#predictions
def predict_match(team_a, team_b):
    features = [
        'Home Team Encoded', 'Away Team Encoded', 'rank_diff', 'points_diff', 'points_ratio',
        'home_top5', 'home_top10', 'home_top20', 'away_top5', 'away_top10', 'away_top20',
        'last_5score_home', 'last_5conceded_home', 'last_5score_away', 'last_5conceded_away',
        'Home_attack_strength', 'Home_defence_strength', 'Away_attack_strength', 'Away_defence_strength',
        'is_group', 'is_knockout', 'is_final', 'h2h_count', 'h2h_avg_goals', 'h2h_home_win_rate'
    ]
    
    match_row = match_data[
        ((match_data['Home Team Name'] == team_a) & (match_data['Away Team Name'] == team_b)) |
        ((match_data['Home Team Name'] == team_b) & (match_data['Away Team Name'] == team_a))
    ]
    
    if len(match_row) == 0:
        team_a_stats = match_data[match_data['Home Team Name'] == team_a].mean(numeric_only=True)
        team_b_stats = match_data[match_data['Home Team Name'] == team_b].mean(numeric_only=True)
        match_row = pd.DataFrame([{
            'Home Team Encoded': team_a_stats.get('Home Team Encoded', 0),
            'Away Team Encoded': team_b_stats.get('Away Team Encoded', 0),
            'rank_diff': team_a_stats.get('rank_diff', 0),
            'points_diff': team_a_stats.get('points_diff', 0),
            'points_ratio': team_a_stats.get('points_ratio', 1),
            'home_top5': 1 if team_a_stats.get('rank_home', 200) <=5 else 0,
            'home_top10': 1 if team_a_stats.get('rank_home', 200) <=10 else 0,
            'home_top20': 1 if team_a_stats.get('rank_home', 200) <=20 else 0,
            'away_top5': 1 if team_b_stats.get('rank_away', 200) <=5 else 0,
            'away_top10': 1 if team_b_stats.get('rank_away', 200) <=10 else 0,
            'away_top20': 1 if team_b_stats.get('rank_away', 200) <=20 else 0,
            'last_5score_home': team_a_stats.get('last_5score_home', 1.2),
            'last_5conceded_home': team_a_stats.get('last_5conceded_home', 1.2),
            'last_5score_away': team_b_stats.get('last_5score_away', 1.2),
            'last_5conceded_away': team_b_stats.get('last_5conceded_away', 1.2),
            'Home_attack_strength': team_a_stats.get('Home_attack_strength', 1),
            'Home_defence_strength': team_a_stats.get('Home_defence_strength', 1),
            'Away_attack_strength': team_b_stats.get('Away_attack_strength', 1),
            'Away_defence_strength': team_b_stats.get('Away_defence_strength', 1),
            'is_group': 0, 'is_knockout': 1, 'is_final': 0,
            'h2h_count': 0, 'h2h_avg_goals': 2.6, 'h2h_home_win_rate': 0.48
        }])
    Xs_2026_scaled = scaler.transform(match_row[features])
    home_goals = np.round(home_model.predict(Xs_2026_scaled)).astype(int)
    away_goals = np.round(away_model.predict(Xs_2026_scaled)).astype(int)
    
    if home_goals > away_goals:
        winner = team_a
    elif away_goals > home_goals:
        winner = team_b
    else:
        # Penalty shootout: slight advantage to higher-ranked team
        rank_a = match_data[match_data['Home Team Name'] == team_a]['rank_home'].mean()
        rank_b = match_data[match_data['Home Team Name'] == team_b]['rank_home'].mean()
        winner = team_a if rank_a < rank_b else team_b
    
    return home_goals, away_goals, winner

#side bar
st.sidebar.title("⚽ 2026 World Cup Hub")
page = st.sidebar.selectbox("Navigate", [
    "🏠 Home",
    "🔮 Match Predictor",
    "👤 Player Dashboard",
    "🏆 Tournament Simulator"
])

#1.home page
if page == "🏠 Home":

    # --- Track popup state ---
    if "popup_closed" not in st.session_state:
        st.session_state.popup_closed = False

    # --- ✅ FIXED TIME & POPUP ---
    if not st.session_state.popup_closed:
        # EAST AFRICAN TIME = UTC+3
        EAT = timezone(timedelta(hours=3))
        # Official kickoff time in EAT
        WC_START = datetime(2026, 6, 11, 22, 0, 0, tzinfo=EAT)

        countdown_placeholder = st.empty()

        # Close button
        col_close = st.columns([4, 1])
        with col_close[1]:
            if st.button("✕", key="close_popup", help="Close countdown popup"):
                st.session_state.popup_closed = True
                st.rerun()

        # Real-time countdown
        while not st.session_state.popup_closed:
            # Get current time in EAT
            now_eat = datetime.now(tz=EAT)
            delta = WC_START - now_eat

            if delta.total_seconds() <= 0:
                days, hours, mins, secs = 0, 0, 0, 0
                break

            # ✅ CORRECT TIME CALCULATION
            days = delta.days
            hours = delta.seconds // 3600
            mins = (delta.seconds // 60) % 60
            secs = delta.seconds % 60

            countdown_placeholder.markdown(f"""
            <div class="popup-overlay">
                <div class="popup-box">
                    <div class="wc-badge">🏆 World Cup 2026</div>
                    <h1 class="countdown-title">THE FINAL COUNTDOWN</h1>
                    <p class="countdown-subtitle">Kick-off to the biggest tournament on the planet</p>
                    <div class="countdown-grid">
                        <div class="countdown-box">
                            <span class="countdown-number">{days:02d}</span>
                            <span class="countdown-label">Days</span>
                        </div>
                        <div class="countdown-box">
                            <span class="countdown-number">{hours:02d}</span>
                            <span class="countdown-label">Hours</span>
                        </div>
                        <div class="countdown-box">
                            <span class="countdown-number">{mins:02d}</span>
                            <span class="countdown-label">Mins</span>
                        </div>
                        <div class="countdown-box">
                            <span class="countdown-number">{secs:02d}</span>
                            <span class="countdown-label">Secs</span>
                        </div>
                    </div>
                    <p class="timezone-note">Time shown in East African Time (EAT, UTC+3)</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(1)


    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.title("🏆 2026 FIFA World Cup Analytics Hub")
    st.markdown("""
    **Your all-in-one platform:**
    - 🤖 Machine Learning modelling for match predictions
    - 👥 Player performance & style clustering
    - 📊 Live interactive dashboards
    - 🏆 Full tournament simulation
    """)

    # --- LOAD NATIONAL TEAMS DATA ---
    @st.cache_data
    def load_teams():
        teams_df = pd.read_csv("R:\\H.H.H\\FIFA  WORLD CUP 2026\\2026 data\\players_info\\national_teams.csv")
       
        qualified_teams = standings['Team']
        teams_df = teams_df[teams_df["name"].isin(qualified_teams)].reset_index(drop=True)
        return teams_df

    teams_df = load_teams()

    st.subheader("🌍 Participating Nations")
    st.divider()

    # --- DISPLAY FLAGS IN GRID (6 per row) ---
    cols = st.columns(6)
    for i, row in teams_df.iterrows():
        with cols[i % 6]:
            st.markdown(f"""
                <div class="flag-card">
                    <img src="{row['team_image_url']}" width="55" class="flag-img"><br>
                    <small>{row['name']}</small>
                </div>
            """, unsafe_allow_html=True)

    st.divider()
    qualified = standings['Team']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Players", len(players_df))
    with col2:
        st.metric("Qualified Teams", len(qualified))
    with col3:
        st.metric("Playing Styles", 6)

    st.markdown('</div>', unsafe_allow_html=True)

#2 Matcg predictor
elif page == "🔮 Match Predictor":
    st.title("🔮 Match Outcome Predictor")
    data_26 = pd.read_csv('R:\\H.H.H\\FIFA  WORLD CUP 2026\\2026 data\\Clean Updated data of 2026.csv')
    # data_26.columns = match_data.columns

    # Fix column name check
    if 'Home Team Name' in data_26.columns:
        home_col = 'Home Team Name'
        away_col = 'Away Team Name'
    elif 'home_team' in data_26.columns:
        home_col = 'home_team'
        away_col = 'away_team'
    else:
        # Use first available team column
        team_cols = [col for col in data_26.columns if 'team' in col.lower() and 'home' in col.lower()]
        home_col = team_cols[0] if team_cols else None
        away_col = home_col.replace('home', 'away') if home_col else None

    if not home_col or not away_col:
        st.error("Team name columns not found in data")
    else:
        all_teams = sorted(pd.concat([data_26[home_col], data_26[away_col]]).dropna().unique())
        col1, col2 = st.columns(2)

        with col1:
            team1 = st.selectbox("Home Team", all_teams)
        with col2:
            team2 = st.selectbox("Away Team", [t for t in all_teams if t != team1])

        if st.button("Predict Result", type="primary"):
            # Get match data
            match_row = data_26[
                ((data_26[home_col] == team1) & (data_26[away_col] == team2)) |
                ((data_26[home_col] == team2) & (data_26[away_col] == team1))
            ]

            features = [
                'Home Team Encoded', 'Away Team Encoded', 'rank_diff', 'points_diff', 'points_ratio',
                'home_top5', 'home_top10', 'home_top20', 'away_top5', 'away_top10', 'away_top20',
                'last_5score_home', 'last_5conceded_home', 'last_5score_away', 'last_5conceded_away',
                'Home_attack_strength', 'Home_defence_strength', 'Away_attack_strength', 'Away_defence_strength',
                'is_group', 'is_knockout', 'is_final', 'h2h_count', 'h2h_avg_goals', 'h2h_home_win_rate'
            ]

            if len(match_row) == 0:
                # Use average stats if no direct match
                team1_stats = data_26[data_26[home_col] == team1].mean(numeric_only=True)
                team2_stats = data_26[data_26[home_col] == team2].mean(numeric_only=True)
                match_row = pd.DataFrame([{
                    'Home Team Encoded': team1_stats.get('Home Team Encoded', 0),
                    'Away Team Encoded': team2_stats.get('Away Team Encoded', 0),
                    'rank_diff': team1_stats.get('rank_diff', 0),
                    'points_diff': team1_stats.get('points_diff', 0),
                    'points_ratio': team1_stats.get('points_ratio', 1),
                    'home_top5': 1 if team1_stats.get('rank_home', 200) <=5 else 0,
                    'home_top10': 1 if team1_stats.get('rank_home', 200) <=10 else 0,
                    'home_top20': 1 if team1_stats.get('rank_home', 200) <=20 else 0,
                    'away_top5': 1 if team2_stats.get('rank_away', 200) <=5 else 0,
                    'away_top10': 1 if team2_stats.get('rank_away', 200) <=10 else 0,
                    'away_top20': 1 if team2_stats.get('rank_away', 200) <=20 else 0,
                    'last_5score_home': team1_stats.get('last_5score_home', 1.2),
                    'last_5conceded_home': team1_stats.get('last_5conceded_home', 1.2),
                    'last_5score_away': team2_stats.get('last_5score_away', 1.2),
                    'last_5conceded_away': team2_stats.get('last_5conceded_away', 1.2),
                    'Home_attack_strength': team1_stats.get('Home_attack_strength', 1),
                    'Home_defence_strength': team1_stats.get('Home_defence_strength', 1),
                    'Away_attack_strength': team2_stats.get('Away_attack_strength', 1),
                    'Away_defence_strength': team2_stats.get('Away_defence_strength', 1),
                    'is_group': 0, 'is_knockout': 1, 'is_final': 0,
                    'h2h_count': 0, 'h2h_avg_goals': 2.6, 'h2h_home_win_rate': 0.48
                }])

            # Predict
            Xs_2026_scaled = scaler.transform(match_row[features])
            home_goals = np.round(home_model.predict(Xs_2026_scaled)).astype(int)
            away_goals = np.round(away_model.predict(Xs_2026_scaled)).astype(int)

            if home_goals > away_goals:
                winner = team1
            elif away_goals > home_goals:
                winner = team2
            else:
                winner = 'Draw'

            st.divider()
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.subheader(team1)
                st.markdown(f"# {home_goals}")

            with col_b:
                st.subheader("VS")
            with col_c:
                st.subheader(team2)
                st.markdown(f"# {away_goals}")

            st.success(f"✅ Predicted Winner: **{winner}**")
       

#3. dashboard for players
elif page == "👤 Player Dashboard":
    st.title("⚽ LIVE PLAYER PERFORMANCE & STYLE ANALYTICS")
    st.markdown("*Clustered by playing style | Data from Transfermarkt*")
    st.divider()

    # --- FILTERS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_positions = st.multiselect(
            "📌 Filter by Position",
            options=sorted(players_df["position"].dropna().unique())
        )
    with col2:
        selected_styles = st.multiselect(
            "🎨 Filter by Playing Style",
            options=sorted(players_df["style_label"].unique())
        )
    with col3:
        selected_clubs = st.multiselect(
            "🏟️ Filter by Club / National Team",
            options=sorted(players_df["club"].dropna().unique())
        )

    # --- APPLY FILTERS ---
    filtered_df = players_df.copy()
    if selected_positions:
        filtered_df = filtered_df[filtered_df["position"].isin(selected_positions)]
    if selected_styles:
        filtered_df = filtered_df[filtered_df["style_label"].isin(selected_styles)]
    if selected_clubs:
        filtered_df = filtered_df[filtered_df["club"].isin(selected_clubs)]

    if filtered_df.empty:
        st.warning("No players match your filters.")
    else:
        # --- MAIN LAYOUT ---
        col_plot, col_profile = st.columns([3, 1.3])

        with col_plot:
            st.subheader("📍 Player Clustering by Style")
            fig_cluster = px.scatter(
                filtered_df,
                x="pca1", y="pca2",
                color="style_label",
                size="market_value",
                size_max=40,
                hover_data={
                    "player_name": True, "club": True, "age": True,
                    "rating": ":,.2f", "goals_per90": ":,.2f", "assists_per90": ":,.2f"
                },
                title="Player Map — Grouped by Playing Style",
                template="plotly_dark"
            )
            fig_cluster.update_layout(height=600, legend=dict(orientation="h", y=1.02, x=1))
            st.plotly_chart(fig_cluster, use_container_width=True)

        with col_profile:
            st.subheader("👤 Player Profile")
            selected_player = st.selectbox("Select a player", sorted(filtered_df["player_name"].unique()))

            if selected_player:
                player = filtered_df[filtered_df["player_name"] == selected_player].iloc[0]
                st.image(player["image_url"], width=220)
                st.markdown(f"### {player['player_name']}")
                st.markdown(f"**{player['style_label']}**")
                st.write(f"🏟️ Club: {player['club']}")
                st.write(f"🎂 Age: {int(player['age'])}")
                st.write(f"⭐ Rating: {player['rating']:.2f}/10")
                st.write(f"💰 Market Value: €{player['market_value']:,}")
                st.divider()
                st.write(f"⚽ Goals/90: {player['goals_per90']:.2f}")
                st.write(f"🎯 Assists/90: {player['assists_per90']:.2f}")
                st.write(f"🔄 Pass Accuracy: {player['pass_accuracy']:.1f}%")
                st.write(f"💨 Dribbles/90: {player['dribbles_per90']:.2f}")
                st.write(f"🛡️ Tackles/90: {player['tackles_per90']:.2f}")

        st.divider()

        # --- BOTTOM CHARTS ---
        col_top, col_radar = st.columns(2)
        with col_top:
            st.subheader("📊 Top Rated Players")
            top10 = filtered_df.sort_values("rating", ascending=False).head(10)
            fig_top = px.bar(top10, x="player_name", y="rating", color="style_label",
                             template="plotly_dark", title="Top 10 Players")
            fig_top.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_top, use_container_width=True)

        with col_radar:
            st.subheader("🕸️ Average Style Profile")
            avg_stats = filtered_df[[
                "goals_per90", "assists_per90", "pass_accuracy",
                "dribbles_per90", "tackles_per90", "interceptions_per90",
                "aerial_duels_won_pct"
            ]].mean()
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_stats.values,
                theta=["Goals/90", "Assists/90", "Pass Accuracy",
                       "Dribbles/90", "Tackles/90", "Interceptions", "Aerial Win %"],
                fill="toself", line_color="#00ff88"
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(range=[-2, 3])), template="plotly_dark")
            st.plotly_chart(fig_radar, use_container_width=True)

#tournament simulation
elif page == "🏆 Tournament Simulator":
    st.title("🏆 2026 World Cup Knockout Stage Simulator")
    
    qualified = standings.head(32)['Team']
    knockout_rounds = ['Round of 32', 'Round of 16', 'Quarter-Final', 'Semi-Final', 'Final']
    
    if st.button("Run Full Simulation", type="primary"):
        remaining = qualified.copy()
        results_log = []
        
        for rnd in knockout_rounds:
            if len(remaining) == 1:
                break
            st.subheader(f"📌 {rnd}")
            next_round = []
            
            for i in range(0, len(remaining), 2):
                if i+1 >= len(remaining):
                    next_round.append(remaining[i])
                    break
                t1, t2 = remaining[i], remaining[i+1]
                hg, ag, winner = predict_match(t1, t2)
                results_log.append(f"{t1} {hg} - {ag} {t2} → Winner: {winner}")
                st.write(f"⚽ {t1} vs {t2} → {hg} - {ag} | ✅ {winner}")
                next_round.append(winner)
            
            remaining = next_round
            st.divider()
        
        st.success(f"🏆 **2026 WORLD CUP CHAMPION: {remaining[0].upper()} 🎉🎉🎉**")

st.sidebar.info('This project was created for the love in football⚽.I hope you find this useful')
# st.sidebar.info('Please don"t forget to share and leaving out your comments. How was this so useful, unuseful, recommendations and to connect')
st.sidebar.info("Your feedback is always welcomed!")
st.sidebar.info("Lets connect! 👇")
st.sidebar.info("Contact: [📧email](mailto:robinsonraphael148@gmail.com)")
st.sidebar.info("WhatsApp: [🔗WhatsApp](https://wa.link/g3kiwf)")
st.sidebar.info("View the Project in GitHub: [💻GitHub](https://github.com/RB148-code)")