import json
import random
import os

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from csp_solver import solve_map_coloring
from groq_helper import explain_solution_groq


st.set_page_config(page_title="CSP Map Coloring (Real-Life)", layout="wide")


@st.cache_data
def load_maps(path: str = "data/maps.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


maps_data = load_maps()

st.title("Map Coloring Problem (CSP) — Real-Life Countries")
st.write("Rule: neighboring countries must have different colors. Solver uses Backtracking + MRV + Forward Checking.")

col1, col2 = st.columns([1, 1])

with col1:
    map_names = list(maps_data.keys())

    choice_mode = st.radio("Map selection", ["Choose", "Random"], horizontal=True)

    if choice_mode == "Random":
        selected_map = random.choice(map_names)
        st.info(f"Random selected: {selected_map}")
    else:
        selected_map = st.selectbox("Select a map/region", map_names)

    color_count = st.slider("Number of colors", min_value=3, max_value=6, value=4)
    palette = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange"]
    colors = palette[:color_count]

    run = st.button("Solve Map Coloring", type="primary")

with col2:
    st.subheader("Groq (optional explanation)")
    use_groq = st.checkbox("Generate explanation using Groq", value=False)
    groq_model = st.text_input("Model", value="llama3-8b-8192")

    # Streamlit Cloud secrets recommended
    groq_key = ""
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
    else:
        groq_key = os.environ.get("GROQ_API_KEY", "")

region = maps_data[selected_map]
countries = region["countries"]
neighbors = region["neighbors"]

st.divider()
st.subheader("Problem Data")
st.write("Countries:", ", ".join(countries))
st.json(neighbors)

if run:
    solution, stats = solve_map_coloring(countries, neighbors, colors)

    st.subheader("Result")
    st.write(
        f"Success: **{stats.success}** | Time: **{stats.elapsed_ms} ms** | "
        f"Assignments: **{stats.assignments}** | Backtracks: **{stats.backtracks}**"
    )

    if not solution:
        st.error("No solution found with given number of colors. Try increasing colors.")
        st.stop()

    # Build graph
    G = nx.Graph()
    for c in countries:
        G.add_node(c)
    for a, nbs in neighbors.items():
        for b in nbs:
            G.add_edge(a, b)

    node_colors = [solution[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200,
