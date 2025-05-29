import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from emergency import EmergencyUnitGA

# Page configuration
st.set_page_config(
    page_title="Emergency Unit Location Optimizer",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #d63384;
#         text-align: center;
#         margin-bottom: 2rem;
#         font-weight: bold;
#     }
#     .section-header {
#         font-size: 1.5rem;
#         color: #0d6efd;
#         margin: 1rem 0;
#         font-weight: bold;
#     }
#     .metric-container {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #0d6efd;
#     }
# </style>
# """, unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🚑 Emergency Unit Location Optimizer</h1>', unsafe_allow_html=True)

# Sidebar for parameters
st.sidebar.header("🔧 GA Parameters")
st.sidebar.markdown("Configure the Genetic Algorithm parameters:")

# GA Parameters
population_size = st.sidebar.slider("Population Size", 20, 100, 50, 10)
generations = st.sidebar.slider("Number of Generations", 50, 200, 100, 10)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.05, 0.3, 0.1, 0.05)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8, 0.1)
city_size = st.sidebar.slider("City Grid Size", 5, 15, 10, 1)

# Run button
run_optimization = st.sidebar.button("🚀 Run Optimization", type="primary")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'ga' not in st.session_state:
    st.session_state.ga = None

# Run optimization when button is clicked
if run_optimization:
    with st.spinner("🔄 Running Genetic Algorithm Optimization..."):
        # Initialize GA
        st.session_state.ga = EmergencyUnitGA(
            city_size=city_size,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        
        # Run evolution
        st.session_state.results = st.session_state.ga.evolve(generations=generations)
    
    st.success("✅ Optimization completed successfully!")

# Display results if available
if st.session_state.results is not None:
    results = st.session_state.results
    ga = st.session_state.ga
    
    # Results summary
    st.markdown('<h2 class="section-header">📊 Optimization Results</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Optimal X Coordinate",
            value=f"{results['best_coordinates'][0]:.3f}",
            help="X coordinate of optimal emergency unit location"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric(
            label="🎯 Optimal Y Coordinate", 
            value=f"{results['best_coordinates'][1]:.3f}",
            help="Y coordinate of optimal emergency unit location"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.metric(
            label="💰 Best Cost Value",
            value=f"{results['best_cost']:.2f}",
            help="Lowest achieved cost function value"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_response = 1.7 + 3.4 * results['average_response_distance']
        st.metric(
            label="⏱️ Avg Response Time",
            value=f"{avg_response:.2f} min",
            help="Average response time to emergencies"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Evolution Chart", "🗺️ City Layout", "📋 Generation Table", "🔥 Emergency Hotspots"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Cost Function Evolution</h3>', unsafe_allow_html=True)
        
        # Create evolution plot
        fig_evolution = go.Figure()
        
        fig_evolution.add_trace(go.Scatter(
            x=results['generation_history'],
            y=results['cost_history'],
            mode='lines+markers',
            name='Best Cost per Generation',
            line=dict(color='#d63384', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Generation:</b> %{x}<br><b>Cost:</b> %{y:.2f}<extra></extra>'
        ))
        
        fig_evolution.update_layout(
            title={
                'text': 'Genetic Algorithm Convergence',
                'x': 0.5,
                'font': {'size': 18, 'color': '#0d6efd'}
            },
            xaxis_title='Generation',
            yaxis_title='Cost Value',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        fig_evolution.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig_evolution.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Add convergence analysis
        if len(results['cost_history']) > 1:
            improvement = results['cost_history'][0] - results['cost_history'][-1]
            improvement_pct = (improvement / results['cost_history'][0]) * 100
            st.info(f"💡 **Convergence Analysis**: Cost improved by {improvement:.2f} units ({improvement_pct:.1f}%) over {generations} generations")
    
    with tab2:
        st.markdown('<h3 class="section-header">Optimal Emergency Unit Location</h3>', unsafe_allow_html=True)
        
        # Create city layout plot
        fig_city = go.Figure()
        
        # Add grid background
        for i in range(city_size + 1):
            fig_city.add_hline(y=i, line_dash="dash", line_color="lightgray", line_width=1)
            fig_city.add_vline(x=i, line_dash="dash", line_color="lightgray", line_width=1)
        
        # Add emergency frequency heatmap
        fig_city.add_trace(go.Heatmap(
            z=results['emergency_frequency_map'],
            x=list(range(city_size)),
            y=list(range(city_size)),
            colorscale='Reds',
            opacity=0.6,
            name='Emergency Frequency',
            hovertemplate='<b>Section:</b> (%{x}, %{y})<br><b>Frequency:</b> %{z:.2f}<extra></extra>',
            showscale=True,
            colorbar=dict(title="Emergency<br>Frequency")
        ))
        
        # Add optimal location
        fig_city.add_trace(go.Scatter(
            x=[results['best_coordinates'][0]],
            y=[results['best_coordinates'][1]],
            mode='markers',
            name='Optimal EU Location',
            marker=dict(
                symbol='star',
                size=20,
                color='blue',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Emergency Unit</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))
        
        # Add evolution path
        x_coords = [coord[0] for coord in results['best_solutions'][::max(1, len(results['best_solutions'])//20)]]
        y_coords = [coord[1] for coord in results['best_solutions'][::max(1, len(results['best_solutions'])//20)]]
        
        fig_city.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            name='Evolution Path',
            line=dict(color='green', width=2, dash='dot'),
            marker=dict(size=4, color='green'),
            opacity=0.7,
            hovertemplate='<b>Evolution Point</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))
        
        fig_city.update_layout(
            title={
                'text': f'{city_size}×{city_size} km City Grid with Optimal Emergency Unit Location',
                'x': 0.5,
                'font': {'size': 16, 'color': '#0d6efd'}
            },
            xaxis_title='X Coordinate (km)',
            yaxis_title='Y Coordinate (km)',
            template='plotly_white',
            height=600,
            xaxis=dict(range=[-0.5, city_size-0.5]),
            yaxis=dict(range=[-0.5, city_size-0.5], scaleanchor="x", scaleratio=1)
        )
        
        st.plotly_chart(fig_city, use_container_width=True)
        
        # Location analysis
        st.info(f"""
        🎯 **Location Analysis**: The optimal emergency unit should be placed at coordinates 
        ({results['best_coordinates'][0]:.3f}, {results['best_coordinates'][1]:.3f}) to achieve 
        minimum response time across the city.
        """)
    
    with tab3:
        st.markdown('<h3 class="section-header">Generation-by-Generation Results</h3>', unsafe_allow_html=True)
        
        # Create and display table
        table_data = ga.get_generation_table()
        df = pd.DataFrame(table_data)
        
        # Show only every 5th generation for readability, plus first and last
        display_df = df.iloc[::5].copy()
        if len(df) > 0 and df.index[-1] not in display_df.index:
            display_df = pd.concat([display_df, df.iloc[[-1]]])
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "Generation": st.column_config.NumberColumn("Gen", width="small"),
                "X Coordinate": st.column_config.NumberColumn("X", format="%.3f"),
                "Y Coordinate": st.column_config.NumberColumn("Y", format="%.3f"),
                "Cost Value": st.column_config.NumberColumn("Cost", format="%.2f"),
                "Avg Response Time (min)": st.column_config.NumberColumn("Response (min)", format="%.2f")
            }
        )
        
        # Download button for full data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results CSV",
            data=csv,
            file_name="emergency_unit_optimization_results.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.markdown('<h3 class="section-header">Emergency Frequency Analysis</h3>', unsafe_allow_html=True)
        
        # Emergency frequency heatmap
        fig_heatmap = px.imshow(
            results['emergency_frequency_map'],
            labels=dict(x="X Coordinate", y="Y Coordinate", color="Emergency Frequency"),
            x=list(range(city_size)),
            y=list(range(city_size)),
            color_continuous_scale="Reds",
            title="Emergency Frequency Distribution Across City Sections"
        )
        
        fig_heatmap.update_layout(
            title={'x': 0.5, 'font': {'size': 16, 'color': '#0d6efd'}},
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Statistics
        freq_stats = {
            "Maximum Frequency": np.max(results['emergency_frequency_map']),
            "Average Frequency": np.mean(results['emergency_frequency_map']),
            "Minimum Frequency": np.min(results['emergency_frequency_map']),
            "Standard Deviation": np.std(results['emergency_frequency_map'])
        }
        
        col1, col2, col3, col4 = st.columns(4)
        for i, (stat, value) in enumerate(freq_stats.items()):
            with [col1, col2, col3, col4][i]:
                st.metric(stat, f"{value:.2f}")

else:
    # Show initial instructions
    st.markdown('<h2 class="section-header">🚀 Get Started</h2>', unsafe_allow_html=True)
    st.info("""
    👈 **Configure parameters** in the sidebar and click "Run Optimization" to begin!
    
    The algorithm will:
    1. Generate random emergency unit locations
    2. Evaluate each location using the cost function
    3. Apply genetic operators (selection, crossover, mutation)
    4. Evolve towards the optimal solution over multiple generations
    """)
    
    # Show sample visualization
    st.markdown('<h3 class="section-header">📊 Sample City Layout</h3>', unsafe_allow_html=True)
    
    # Create sample emergency frequency map
    sample_freq = np.random.exponential(scale=2.0, size=(10, 10))
    sample_freq = sample_freq / np.max(sample_freq) * 10
    
    fig_sample = px.imshow(
        sample_freq,
        labels=dict(x="X Coordinate", y="Y Coordinate", color="Emergency Frequency"),
        color_continuous_scale="Reds",
        title="Sample Emergency Frequency Distribution (10×10 km City)"
    )
    
    fig_sample.update_layout(
        title={'x': 0.5, 'font': {'size': 16, 'color': '#0d6efd'}},
        height=400
    )
    
    st.plotly_chart(fig_sample, use_container_width=True)