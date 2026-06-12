import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

#Loading the data
df = pd.read_csv("transfermarkt_players_clean.csv")


# Clean & fix images
def fix_image_url(url):
    if pd.isna(url) or str(url) == "" or str(url) == "nan":
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/400px-No-Image-Placeholder.svg.png"
    if not str(url).startswith("http"):
        return f"https://img.a.transfermarkt.technology/{str(url).lstrip('/')}"
    return url

df["image_url"] = df["image_url"].apply(fix_image_url)
df = df.dropna(subset=["player_name", "position", "rating"])

print(f"✅ Loaded {len(df)} players")


# Features that show how players play
style_features = [
    "goals_per90", "assists_per90",
    "pass_accuracy", "progressive_passes",
    "dribbles_per90", "duels_won_pct",
    "tackles_per90", "interceptions_per90",
    "aerial_duels_won_pct"
]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[style_features])

# Cluster → 6 standard football styles
kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
df["style_cluster"] = kmeans.fit_predict(X_scaled)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
df[["pca1", "pca2"]] = pca.fit_transform(X_scaled)

# Name clusters based on their stats
cluster_profile = df.groupby("style_cluster")[style_features].mean()

style_names = {
    0: "⚽ FINISHING FORWARD",      # High goals, low defense
    1: "🎯 CREATIVE PLAYMAKER",     # High passes, assists
    2: "🛡️ DEFENSIVE ANCHOR",       # High tackles, interceptions
    3: "💨 DYNAMIC WINGER",         # High dribbles, duels
    4: "🧱 BALL-PLAYING DEFENDER",  # Good passes + defense
    5: "🔄 BOX-TO-BOX MIDFIELDER"   # All-around, balanced
}

df["style_label"] = df["style_cluster"].map(style_names)

# Dashboard 
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Player Performance Dashboard"
)

#layout
app.layout = dbc.Container([

    # HEADER
    html.Div([
        html.H1("⚽ LIVE PLAYER PERFORMANCE & STYLE ANALYTICS",
                className="text-center text-primary mb-2"),
        html.P("Clustered by playing style | Data from Transfermarkt | Updated every 5 min",
               className="text-center text-muted")
    ]),

    html.Hr(),

    # FILTERS
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id="pos_filter",
                options=[{"label": p, "value": p} for p in sorted(df["position"].unique())],
                multi=True,
                placeholder="📌 Filter by Position",
                className="mb-2"
            )
        ], width=4),
        dbc.Col([
            dcc.Dropdown(
                id="style_filter",
                options=[{"label": s, "value": s} for s in style_names.values()],
                multi=True,
                placeholder="🎨 Filter by Playing Style",
                className="mb-2"
            )
        ], width=4),
        dbc.Col([
            dcc.Dropdown(
                id="club_filter",
                options=[{"label": c, "value": c} for c in sorted(df["club"].unique())],
                multi=True,
                placeholder="🏟️ Filter by Club / National Team",
                className="mb-2"
            )
        ], width=4)
    ]),

    html.Hr(),

    # MAIN CONTENT: CLUSTER PLOT + PLAYER PROFILE
    dbc.Row([
        # LEFT: Interactive Cluster Visualization
        dbc.Col([
            html.H4("📍 Player Clustering by Style", className="text-center"),
            dcc.Graph(
                id="cluster_plot",
                config={"displayModeBar": True, "scrollZoom": True},
                style={"height": "650px"}
            )
        ], width=8),

        # RIGHT: PLAYER CARD WITH PHOTO
        dbc.Col([
            html.H4("👤 Player Profile", className="text-center"),
            html.Div(
                id="player_card",
                children=[
                    html.P("Click any dot on the map to view player details",
                           className="text-center text-muted mt-5"),
                    html.Img(
                        src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/400px-No-Image-Placeholder.svg.png",
                        style={
                            "width": "240px", "height": "280px",
                            "objectFit": "cover", "borderRadius": "12px",
                            "border": "3px solid #555", "display": "block",
                            "margin": "30px auto"
                        }
                    )
                ],
                style={
                    "padding": "20px", "border": "1px solid #444",
                    "borderRadius": "12px", "height": "650px",
                    "overflowY": "auto"
                }
            )
        ], width=4)
    ]),

    html.Hr(),

    # BOTTOM: TOP PLAYERS & STYLE PROFILE
    dbc.Row([
        dbc.Col([
            html.H4("📊 Top Rated Players", className="text-center"),
            dcc.Graph(id="top_players_chart")
        ], width=6),
        dbc.Col([
            html.H4("🕸️ Average Style Profile", className="text-center"),
            dcc.Graph(id="radar_chart")
        ], width=6)
    ]),

    # AUTO REFRESH EVERY 5 MINUTES
    dcc.Interval(
        id="live_update",
        interval=300000,  # 5 minutes
        n_intervals=0
    )

], fluid=True)


# Update Cluster Plot
@app.callback(
    Output("cluster_plot", "figure"),
    Input("pos_filter", "value"),
    Input("style_filter", "value"),
    Input("club_filter", "value"),
    Input("live_update", "n_intervals")
)
def update_cluster(pos, style, club, n):
    dff = df.copy()
    if pos:
        dff = dff[dff["position"].isin(pos)]
    if style:
        dff = dff[dff["style_label"].isin(style)]
    if club:
        dff = dff[dff["club"].isin(club)]

    fig = px.scatter(
        dff,
        x="pca1", y="pca2",
        color="style_label",
        size="market_value",
        size_max=40,
        hover_data={
            "player_name": True,
            "club": True,
            "age": True,
            "rating": True,
            "goals_per90": ":,.2f",
            "assists_per90": ":,.2f"
        },
        custom_data=[
            "player_name", "club", "age", "rating",
            "image_url", "style_label", "market_value",
            "goals_per90", "assists_per90", "pass_accuracy",
            "dribbles_per90", "tackles_per90"
        ],
        title="Player Map — Clustered by Playing Style",
        template="plotly_dark"
    )

    fig.update_layout(
        clickmode="event+select",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# Show Player Details + PHOTO when clicked
@app.callback(
    Output("player_card", "children"),
    Input("cluster_plot", "clickData")
)
def show_player(clickData):
    if not clickData:
        return [
            html.P("👆 Click any player dot", className="text-muted"),
            html.Img(
                src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/400px-No-Image-Placeholder.svg.png",
                style={"width": "220px", "borderRadius": "10px", "margin": "30px auto", "display": "block"}
            )
        ]

    # Extract data
    cd = clickData["points"][0]["customdata"]
    name, club, age, rating, img, style, mv, g90, a90, pass_acc, drib, tack = cd

    return [
        html.H3(name, className="text-primary text-center"),
        html.P(f"🏟️ {club} | 🎂 {age} yrs", className="text-center"),
        html.P(f"⭐ Rating: {rating}/10 | 💰 Value: €{mv:,}", className="text-center text-info"),

        html.Img(src=img, className="player-photo"),

        html.H5(f"🎨 {style}", className="text-center text-success mt-3"),
        html.Hr(),

        html.P(f"⚽ Goals/90: {g90:.2f}", className="mb-1"),
        html.P(f"🎯 Assists/90: {a90:.2f}", className="mb-1"),
        html.P(f"🔄 Pass Accuracy: {pass_acc:.1f}%", className="mb-1"),
        html.P(f"💨 Dribbles/90: {drib:.2f}", className="mb-1"),
        html.P(f"🛡️ Tackles/90: {tack:.2f}", className="mb-1")
    ]

# Top Players Chart
@app.callback(
    Output("top_players_chart", "figure"),
    Input("pos_filter", "value"),
    Input("style_filter", "value"),
    Input("club_filter", "value")
)
def top_players(pos, style, club):
    dff = df.copy()
    if pos: dff = dff[dff["position"].isin(pos)]
    if style: dff = dff[dff["style_label"].isin(style)]
    if club: dff = dff[dff["club"].isin(club)]

    top10 = dff.sort_values("rating", ascending=False).head(10)

    fig = px.bar(
        top10,
        x="player_name",
        y="rating",
        color="style_label",
        hover_data=["club", "market_value"],
        title="Highest Rated Players",
        template="plotly_dark"
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

# Radar Chart: Style Profile
@app.callback(
    Output("radar_chart", "figure"),
    Input("pos_filter", "value"),
    Input("style_filter", "value"),
    Input("club_filter", "value")
)
def radar(pos, style, club):
    dff = df.copy()
    if pos: dff = dff[dff["position"].isin(pos)]
    if style: dff = dff[dff["style_label"].isin(style)]
    if club: dff = dff[dff["club"].isin(club)]

    avg_stats = dff[style_features].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_stats.values,
        theta=[f.replace("_", " ").upper() for f in style_features],
        fill="toself",
        name="Average Profile",
        line_color="#00ff88"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2, 3])),
        template="plotly_dark",
        title="Statistical Style Profile"
    )
    return fig

#Main 
if __name__ == "__main__":
    app.run(debug=True, port=8050)
    