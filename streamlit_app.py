
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from backend.sim import simulate_netlist, _simulate_deterministic
from backend.features import extract_features, compute_scatter
from backend.ml import train_svm, predict_fault, load_model
from backend.demo_data import generate_training_dataset, generate_per_class_waveforms
from backend.utils import inject_fault
import time
import os

# Page Config
st.set_page_config(
    page_title="RLC Fault Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #0f172a;
        color: white;
    }
    .metric-card {
        background-color: #f1f5f9;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Constants
FAULT_CLASSES = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']
CLASS_LABELS = {
    'E0': 'Healthy',
    'E1': 'R1 Open',
    'E2': 'R2 Open',
    'E3': 'L1 Fault',
    'E4': 'C1 Fault',
    'E5': 'Short'
}
CLASS_COLORS = {
    'E0': '#10b981', # green
    'E1': '#3b82f6', # blue
    'E2': '#8b5cf6', # purple
    'E3': '#f59e0b', # amber
    'E4': '#ef4444', # red
    'E5': '#ec4899', # pink
}

def main():
    st.title("⚡ RLC Circuit Fault Detection")
    st.markdown("### SVM-Based Classification using Phasor Features")
    
    # --- Sidebar: Input ---
    with st.sidebar:
        st.header("Circuit Configuration")
        default_netlist = """V1 in 0 SIN(0 10 50)
R1 in 1 1000
R2 1 2 1000
L1 1 0 2
C1 2 0 6u
.tran 20m 0 0.01m
.END"""
        netlist = st.text_area("Netlist (SPICE format)", value=default_netlist, height=200)
        
        analyze_btn = st.button("Analyze Circuit", type="primary")
        
        st.markdown("---")
        st.markdown("**System Status**")
        
        # Check if model exists
        model_path = os.path.join("backend", "models", "svm_model.joblib")
        if os.path.exists(model_path):
            st.success("Model Loaded")
        else:
            st.warning("Model Not Found (Will Retrain)")

    # --- Main Logic ---
    if analyze_btn:
        with st.spinner("Simulating and Classifying..."):
            # 1. Simulate & Predict
            try:
                # Ensure model exists
                if not os.path.exists(model_path):
                    status_text = st.empty()
                    status_text.info("Training new model...")
                    X, y, dist_data = generate_training_dataset(netlist)
                    train_svm(X, y)
                    status_text.empty()
                
                # Run Simulation for current netlist
                sim_output = simulate_netlist(netlist)
                features = extract_features(sim_output)
                
                # Predict
                prediction, probs = predict_fault(features)
                predicted_label = CLASS_LABELS.get(prediction, prediction)
                
                # Generate Waveforms for all classes (for comparison)
                waveforms = generate_per_class_waveforms(netlist)
                
                # Load Metrics
                _, metrics = load_model()
                
                # --- Display Results ---
                
                # Top Banner: Prediction
                color = CLASS_COLORS.get(prediction, "#333")
                st.markdown(f"""
                <div style="background-color: {color}20; border: 2px solid {color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h2 style="color: {color}; margin:0;">Fault Detected: {prediction} ({predicted_label})</h2>
                    <p style="margin:0;">Confidence: {max(probs)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tabs for Details
                tab1, tab2, tab3 = st.tabs(["📈 Waveforms", "🎯 Distribution", "📊 Metrics"])
                
                with tab1:
                    st.subheader("Circuit Response")
                    
                    # Class Selector
                    selected_class = st.selectbox("Compare with Class:", FAULT_CLASSES, index=FAULT_CLASSES.index(prediction))
                    
                    # Plotly Waveforms
                    fig = go.Figure()
                    
                    # Input
                    fig.add_trace(go.Scatter(x=waveforms['t'], y=waveforms['vin'], name='Input (Vin)', line=dict(color='gray', dash='dash')))
                    
                    # V0
                    v0_key = f"v0_{selected_class}"
                    if v0_key in waveforms:
                        fig.add_trace(go.Scatter(x=waveforms['t'], y=waveforms[v0_key], name=f'V0 ({selected_class})', line=dict(color=CLASS_COLORS[selected_class])))
                    
                    # V1
                    v1_key = f"v1_{selected_class}"
                    if v1_key in waveforms:
                        fig.add_trace(go.Scatter(x=waveforms['t'], y=waveforms[v1_key], name=f'V1 ({selected_class})', line=dict(color=CLASS_COLORS[selected_class], dash='dot')))
                        
                    fig.update_layout(
                        title=f"Waveforms for {selected_class} ({CLASS_LABELS[selected_class]})",
                        xaxis_title="Time (s)",
                        yaxis_title="Voltage (V)",
                        height=400,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    st.subheader("Phasor Distribution (V0)")
                    
                    # Get distribution data (regenerate if needed or load from somewhere? 
                    # For now, let's generate a small set or use what we have. 
                    # Ideally we should cache this. Let's regenerate for visualization.)
                    
                    # We can use the training data from the model training step if we saved it, 
                    # but simpler to just generate a few points for visualization.
                    if 'dist_data' not in st.session_state:
                         _, _, st.session_state.dist_data = generate_training_dataset(netlist, N=10)
                    
                    dist_data = st.session_state.dist_data
                    
                    fig_dist = go.Figure()
                    
                    for cls in FAULT_CLASSES:
                        if cls in dist_data:
                            points = np.array(dist_data[cls])
                            fig_dist.add_trace(go.Scatter(
                                x=points[:, 0], 
                                y=points[:, 1], 
                                mode='markers',
                                name=f"{cls} ({CLASS_LABELS[cls]})",
                                marker=dict(color=CLASS_COLORS[cls], size=8, opacity=0.7)
                            ))
                            
                    # Highlight current prediction
                    current_v0_real = features[0]
                    current_v0_imag = features[1]
                    fig_dist.add_trace(go.Scatter(
                        x=[current_v0_real],
                        y=[current_v0_imag],
                        mode='markers',
                        name='Current Simulation',
                        marker=dict(color='black', size=15, symbol='x')
                    ))
                    
                    fig_dist.update_layout(
                        title="V0 Phasor Distribution (Real vs Imaginary)",
                        xaxis_title="Real Part (V)",
                        yaxis_title="Imaginary Part (V)",
                        height=500
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                with tab3:
                    st.subheader("Model Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Overall Accuracy", f"{metrics['accuracy']*100:.1f}%")
                        
                    with col2:
                        # Confusion Matrix Heatmap
                        cm = np.array(metrics['confusion'])
                        fig_cm = px.imshow(
                            cm,
                            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                            x=FAULT_CLASSES,
                            y=FAULT_CLASSES,
                            text_auto=True,
                            color_continuous_scale="Mint"
                        )
                        fig_cm.update_layout(title="Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    else:
        st.info("👈 Click 'Analyze Circuit' to start simulation.")
        
        # Show placeholder metrics if model exists
        model_path = os.path.join("backend", "models", "svm_model.joblib")
        if os.path.exists(model_path):
             _, metrics = load_model()
             if metrics:
                st.markdown("### Current Model Metrics")
                cm = np.array(metrics['confusion'])
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    x=FAULT_CLASSES,
                    y=FAULT_CLASSES,
                    text_auto=True,
                    color_continuous_scale="Mint"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

if __name__ == "__main__":
    main()
