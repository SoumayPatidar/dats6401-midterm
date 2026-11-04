# DATS 6401: Mid-Term Project
# Soumay Patidar
# Formula 1 World Championship: Multi-Modality Visualization
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations

st.set_page_config(page_title="F1 Dashboard", page_icon="🏎️", layout="wide")

#DATA LOADING (cached)
@st.cache_data
def load_all():
    results = pd.read_csv("data/results.csv")
    races = pd.read_csv("data/races.csv")
    drivers = pd.read_csv("data/drivers.csv")
    constructors = pd.read_csv("data/constructors.csv")
    circuits = pd.read_csv("data/circuits.csv")
    status = pd.read_csv("data/status.csv")

    df = (
        results
        .merge(races[["raceId", "year", "date", "name", "circuitId"]], on="raceId")
        .merge(drivers[["driverId", "forename", "surname", "nationality"]], on="driverId")
        .merge(constructors[["constructorId", "name"]], on="constructorId",
               suffixes=("_race", "_constructor"))
        .merge(status[["statusId", "status"]], on="statusId")
    )
    df["driver"] = df["forename"] + " " + df["surname"]
    df["date"] = pd.to_datetime(df["date"])
    df["position_num"] = pd.to_numeric(df["position"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    return df, circuits

df, circuits = load_all()

#SIDEBAR
st.sidebar.title("🏎️ F1 Dashboard")
st.sidebar.markdown("**Mid-Term Project**")

year_range = st.sidebar.slider(
    "Year range",
    int(df["year"].min()), int(df["year"].max()),
    (1990, 2024),
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **About this app**

    This dashboard explores 70+ years of Formula 1 
    data across four modalities: time series trends, 
    driver teammate networks, a world map of circuits, 
    and multivariate analysis of driver careers.

    Data from the Ergast F1 Database.
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("DATS 6401 - Soumay Patidar")

# filter
filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

# HEADER
st.title("🏎️ Formula 1: Multi-Modality Visualization")
st.markdown(
    f"""
    Exploring F1 data from **{year_range[0]}** to **{year_range[1]}** across 
    four different data modalities. Use the sidebar to change the year range, all tabs update automatically.
    """
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Seasons", year_range[1] - year_range[0] + 1)
c2.metric("Races", f"{filtered['raceId'].nunique():,}")
c3.metric("Drivers", f"{filtered['driverId'].nunique():,}")
c4.metric("Circuits", f"{filtered['circuitId'].nunique():,}")

st.markdown("---")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    " Time Series",
    " Network",
    "️ Map",
    " Multivariate",
])


#
# TAB 1: TIME SERIES
#
with tab1:
    st.markdown("### Season Trends Over Time")
    st.markdown(
        "Looking at how F1 has grown and changed over the decades. "
        "The year range filter in the sidebar controls what's shown here."
    )

    # yearly stats
    yearly = filtered.groupby("year").agg(
        total_points=("points", "sum"),
        num_races=("raceId", "nunique"),
        num_drivers=("driverId", "nunique"),
    ).reset_index()

    # which metric to plot
    ts_metric = st.selectbox(
        "Metric to plot",
        ["num_races", "total_points", "num_drivers"],
        format_func=lambda x: {
            "num_races": "Races per season",
            "total_points": "Total points awarded",
            "num_drivers": "Unique drivers",
        }[x],
    )

    fig1 = px.line(
        yearly, x="year", y=ts_metric,
        title=f"F1 {ts_metric.replace('_', ' ').title()} by Season",
        labels={"year": "Season", ts_metric: ts_metric.replace("_", " ").title()},
    )
    fig1.update_traces(line_color="steelblue", line_width=2)
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # top constructors over time
    st.markdown("### Constructor Points by Season")
    constructor_pts = (
        filtered.groupby(["year", "name_constructor"])["points"]
        .sum().reset_index()
    )
    top_teams = (
        constructor_pts.groupby("name_constructor")["points"]
        .sum().nlargest(6).index.tolist()
    )
    constructor_pts = constructor_pts[constructor_pts["name_constructor"].isin(top_teams)]

    fig2 = px.line(
        constructor_pts, x="year", y="points", color="name_constructor",
        title="Top 6 Constructors: Points per Season",
        labels={"year": "Season", "points": "Points", "name_constructor": "Team"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        **What this shows:** You can see how the sport has expanded: 
        more races per season, more points being awarded, more drivers 
        competing. The constructor chart shows eras of dominance, 
        Ferrari in the early 2000s, Red Bull around 2010-2013, 
        Mercedes from 2014 onward.

        **Caveat:** The points system has changed multiple times 
        (most recently in 2010 when a win went from 10 to 25 points), 
        so raw point totals across decades aren't directly comparable. 
        The jumps you see partly reflect rule changes, not just 
        performance changes.
        """
    )


#
# TAB 2: NETWORK
#
with tab2:
    st.markdown("### Driver Teammate Network")
    st.markdown(
        "Two drivers are connected if they raced for the same team in "
        "the same season. Drivers who moved between many teams become "
        "hubs connecting different parts of the network."
    )

    min_degree = st.slider("Min connections to show a driver", 1, 8, 2)

    @st.cache_data
    def build_network(start, end):
        sub = df[(df["year"] >= start) & (df["year"] <= end)]
        teams = sub.groupby(["year", "name_constructor"])["driver"].apply(set).reset_index()
        teams = teams[teams["driver"].apply(len) >= 2]

        edges = []
        for _, row in teams.iterrows():
            for d1, d2 in combinations(sorted(row["driver"]), 2):
                edges.append({"driver1": d1, "driver2": d2})

        if not edges:
            return None

        edge_df = pd.DataFrame(edges)
        agg = edge_df.groupby(["driver1", "driver2"]).size().reset_index(name="weight")
        G = nx.from_pandas_edgelist(agg, "driver1", "driver2", ["weight"])
        return G

    G = build_network(year_range[0], year_range[1])

    if G and G.number_of_nodes() > 0:
        # filter by degree
        keep = [n for n in G.nodes() if G.degree(n) >= min_degree]
        G_sub = G.subgraph(keep).copy()

        # communities
        comms = list(greedy_modularity_communities(G_sub))
        comm_map = {}
        for cid, comm in enumerate(comms):
            for node in comm:
                comm_map[node] = cid

        nc1, nc2, nc3 = st.columns(3)
        nc1.metric("Drivers shown", G_sub.number_of_nodes())
        nc2.metric("Teammate links", G_sub.number_of_edges())
        nc3.metric("Communities", len(comms))

        # colors
        palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                    "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
                    "#e6beff", "#9A6324", "#800000", "#aaffc3"]

        degrees = dict(G_sub.degree())
        max_deg = max(degrees.values()) if degrees else 1

        net = Network(height="500px", width="100%", cdn_resources="in_line")
        for n in G_sub.nodes():
            size = 10 + (degrees[n] / max_deg) * 35
            color = palette[comm_map.get(n, 0) % len(palette)]
            net.add_node(n, label=n.split()[-1], size=size, color=color,
                         title=f"{n}\nTeammates: {degrees[n]}\nCommunity: {comm_map.get(n, '?')}")
        for u, v, d in G_sub.edges(data=True):
            net.add_edge(u, v, value=d.get("weight", 1), color="#cccccc")

        html = net.generate_html(notebook=False)
        components.html(html, height=520)

        st.caption(
            "Node size = number of different teammates · "
            "Node color = detected community · Hover for details"
        )

        st.markdown(
            """
            **What this shows:** Drivers who drove for many different teams 
            (like Ricciardo or Hulkenberg) end up as hubs with lots of 
            connections. The colored communities roughly correspond to 
            "team families": groups of drivers who circulated among the 
            same constructors. 

            **Caveat:** The layout positions are arbitrary, the 
            force-directed algorithm places connected nodes closer, 
            but x/y doesn't mean anything real. Also, with too many 
            years selected, it becomes a hairball where you can't 
            read anything.
            """
        )
    else:
        st.warning("Not enough data for the selected year range.")


#
# TAB 3: MAP
#
with tab3:
    st.markdown("### F1 Circuits Around the World")
    st.markdown(
        "Every circuit that hosted a race in the selected year range, "
        "plotted on a world map."
    )

    # which circuits were used in the filtered data
    used_circuits = filtered["circuitId"].unique()
    map_data = circuits[circuits["circuitId"].isin(used_circuits)].copy()

    # count races per circuit
    race_counts = (
        filtered.groupby("circuitId")["raceId"]
        .nunique().reset_index()
        .rename(columns={"raceId": "num_races"})
    )
    map_data = map_data.merge(race_counts, on="circuitId")

    st.markdown(f"**{len(map_data)} circuits** across **{map_data['country'].nunique()} countries**")

    fig3 = px.scatter_geo(
        map_data,
        lat="lat", lon="lng",
        size="num_races",
        color="country",
        hover_name="name",
        hover_data={"location": True, "country": True, "num_races": True,
                     "lat": False, "lng": False},
        title="F1 Circuit Locations",
        size_max=20,
    )
    fig3.update_layout(
        height=500,
        showlegend=False,
        geo=dict(
            showland=True, landcolor="lightgray",
            showocean=True, oceancolor="aliceblue",
            showcountries=True, countrycolor="white",
        ),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # top countries
    st.markdown("### Races by Country")
    country_races = (
        filtered.merge(circuits[["circuitId", "country"]], on="circuitId")
        .groupby("country")["raceId"].nunique()
        .sort_values(ascending=True).tail(15)
    )

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    country_races.plot.barh(ax=ax4, color="steelblue")
    ax4.set_xlabel("Number of races hosted")
    ax4.set_title("Top 15 Countries by Races Hosted")
    plt.tight_layout()
    st.pyplot(fig4)

    st.markdown(
        """
        **What this shows:** F1 started as a mostly European sport: the 
        map makes this really obvious with the cluster of dots in Western 
        Europe. But over time it's expanded to the Middle East, Asia, 
        and the Americas. Bubble size shows how many races each circuit 
        hosted, bigger bubbles like Monza and Silverstone have been on 
        the calendar for decades.

        **Caveat:** The map uses scatter points, not regions, so countries 
        with multiple circuits (like the US with Austin, Miami, Las Vegas) 
        might look more "covered" than they really are. Also, a circuit 
        that hosted 1 race and one that hosted 50 look similar unless you 
        check the bubble size carefully.
        """
    )


#
# TAB 4: MULTIVARIATE
#
with tab4:
    st.markdown("### Driver Career Stats: Correlation & PCA")
    st.markdown(
        "Aggregating race results into per-driver career statistics and "
        "looking for patterns across multiple dimensions at once."
    )

    min_races = st.slider("Min career races to include", 10, 100, 30)

    @st.cache_data
    def build_stats(data, min_r):
        stats = data.groupby(["driver", "nationality"]).agg(
            total_races=("raceId", "nunique"),
            total_wins=("position_num", lambda x: (x == 1).sum()),
            total_podiums=("position_num", lambda x: (x <= 3).sum()),
            total_points=("points", "sum"),
            avg_grid=("grid", "mean"),
            avg_finish=("position_num", "mean"),
            dnf_count=("status", lambda x: (x != "Finished").sum()),
        ).reset_index()
        stats = stats[stats["total_races"] >= min_r].reset_index(drop=True)
        stats["win_rate"] = round(stats["total_wins"] / stats["total_races"] * 100, 2)
        stats["podium_rate"] = round(stats["total_podiums"] / stats["total_races"] * 100, 2)
        return stats

    driver_stats = build_stats(filtered, min_races)
    numeric_cols = [
        "total_races", "total_wins", "total_podiums", "total_points",
        "avg_grid", "avg_finish", "dnf_count", "win_rate", "podium_rate",
    ]

    st.markdown(f"**{len(driver_stats)} drivers** with {min_races}+ races")

    mv_col1, mv_col2 = st.columns(2)

    with mv_col1:
        # correlation heatmap
        fig5, ax5 = plt.subplots(figsize=(7, 6))
        corr = driver_stats[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                    square=True, ax=ax5, cbar_kws={"shrink": 0.7},
                    annot_kws={"size": 7})
        ax5.set_title("Correlation Heatmap", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig5)

    with mv_col2:
        # PCA
        color_pca = st.selectbox(
            "Color PCA by",
            ["nationality", "total_wins"],
            format_func=lambda x: "Nationality" if x == "nationality" else "Win count",
        )

        X = driver_stats[numeric_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": pcs[:, 0], "PC2": pcs[:, 1],
            "driver": driver_stats["driver"],
            "color": driver_stats[color_pca],
        })

        fig6 = px.scatter(
            pca_df, x="PC1", y="PC2", color="color",
            hover_data=["driver"],
            title=f"PCA - PC1: {pca.explained_variance_ratio_[0]:.0%}, PC2: {pca.explained_variance_ratio_[1]:.0%}",
            labels={"color": color_pca.replace("_", " ").title()},
        )
        fig6.update_layout(height=500)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown(
        """
        **What this shows:** The heatmap confirms what you'd expect:
        wins, podiums, and points are all highly correlated. Average 
        grid and average finish are correlated too (qualify well, finish 
        well). The PCA scatter compresses all 9 dimensions into 2 and 
        shows that most of the variance comes from a "career success" 
        axis (PC1): the legends like Hamilton and Schumacher are way 
        out on the right, while mid-field drivers cluster together.

        **Caveat:** PCA axes don't have simple meanings: PC1 is a 
        weighted combination of all 9 variables, not just "wins." Also 
        standardization matters: without it, variables with bigger 
        numbers (like total_points) would dominate just because of scale, 
        not because they're more important.
        """
    )
