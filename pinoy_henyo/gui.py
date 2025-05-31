import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from game_master import GameMaster
from guesser2 import Guesser

st.set_page_config(page_title="Pinoy Henyo", page_icon="🎯", layout="wide")

st.title("🎯 Pinoy Henyo - Word Guessing Game")
st.markdown("---")

# Initialize session state
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Center input section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### Enter the Secret Word")
    secret_word = st.text_input("", placeholder="Type your word here...", key="word_input")
    
    if st.button("🚀 Start Guessing!", type="primary", use_container_width=True):
        if secret_word.strip():
            st.session_state.game_started = True
            st.session_state.secret_word = secret_word.lower().strip()
        else:
            st.error("Please enter a word first!")

# Game execution
if st.session_state.game_started and 'secret_word' in st.session_state:
    secret = st.session_state.secret_word
    
    # Create game master and guesser
    game_master = GameMaster(secret)
    guesser = Guesser(len(secret))
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Centered live table
    live_progress_container = st.container()
    with live_progress_container:
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.markdown("### 🔄 Live Progress")
        live_table = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    
    generations_data = []
    console_logs = []
    
    # Run genetic algorithm
    for generation in range(guesser.max_generations):
        # Update progress
        progress = (generation + 1) / guesser.max_generations
        progress_bar.progress(progress)
        status_text.text(f"Generation {generation + 1}/{guesser.max_generations} - Processing...")
        
        # Get best guess
        best_guess = guesser.get_best_individual()
        cost = game_master.calculate_cost(best_guess)
        
        # Store data
        generations_data.append({
            'Generation': generation + 1,
            'Best Guess': best_guess,
            'Cost Value': cost
        })
        
        # Update live table (show last 5 generations)
        display_data = generations_data[-5:] if len(generations_data) > 5 else generations_data
        live_df = pd.DataFrame(display_data)
        live_table.dataframe(live_df, use_container_width=True, height=200)

    
        
        
        # Evolve to next generation
        if generation < guesser.max_generations - 1:
            guesser.evolve_generation(game_master)
        
    
    # Final results
    final_guess = generations_data[-1]['Best Guess']
    final_cost = generations_data[-1]['Cost Value']
    total_generations = len(generations_data)
    
    # Hide progress and live table
    progress_bar.empty()
    status_text.empty()
    live_table.empty()
    live_progress_container.empty()
    
    st.markdown("---")
    
    # Success/Failure modal
    if final_cost == 0:
        st.success("🎉 Successfully Guessed!")
        st.balloons()
    else:
        st.error("❌ Could not guess perfectly")
    
    # Final Results Section with Grid Layout
    st.markdown("### 🏆 Final Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Final cost",
            value=final_cost,
        )
    
    with col2:
        st.metric(
            label="⭐ Secret Word",
            value=secret.upper()
        )
    
    with col3:
        st.metric(
            label="💡 Best Guess",
            value=final_guess.upper()
        )
    
    with col4:
        st.metric(
            label="🎲 Match Status",
            value="Perfect match!" if final_cost == 0 else "Close Match",
            delta=f"Cost difference: {final_cost}"
        )
    
    st.markdown("---")
    
    # Results table and chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Generation Results")
        df = pd.DataFrame(generations_data)
        st.dataframe(df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 📈 Cost Value vs Generation")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Generation'],
            y=df['Cost Value'],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=5, color='#ff7f0e'),
            name='Cost Value',
            hovertemplate='Generation: %{x}<br>Cost: %{y}<extra></extra>'
        ))
        fig.update_layout(
            xaxis_title="Generation",
            yaxis_title="Cost Value",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Reset button
    if st.button("🔄 Try Another Word", type="secondary", use_container_width=True):
        st.session_state.game_started = False
        st.session_state.results = None
        if 'secret_word' in st.session_state:
            del st.session_state.secret_word
        st.rerun()