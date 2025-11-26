import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import time

# Page config
st.set_page_config(page_title="Loss Landscape 3D", layout="wide", page_icon="🏔️")

# Custom CSS for beautiful dark theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .stApp {
        background: transparent;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏔️ Real-Time Loss Landscape Explorer")
st.markdown("### Visualize the 3D terrain of your model's optimization journey")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (s)", 1, 30, 5)

if st.sidebar.button("🔄 Refresh Now") or auto_refresh:
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# Load training history
@st.cache_data
def load_training_data():
    log_dir = Path("../logs")
    history_file = log_dir / "training_history.json"
    
    if not history_file.exists():
        return None, "No training data found. Start training first!"
        
    with open(history_file, 'r') as f:
        data = json.load(f)
        
    return data, None

data, error = load_training_data()

if error:
    st.error(error)
    st.stop()

# Extract data
epochs = list(range(1, len(data['train_losses']) + 1))
train_losses = data['train_losses']
val_losses = data['val_losses']
learning_rates = data.get('learning_rates', [1e-4] * len(epochs))

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Current Epoch", len(epochs), f"{len(epochs)}/1000")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Train Loss", f"{train_losses[-1]:.4f}", 
              f"{train_losses[-1] - train_losses[-2]:.4f}" if len(train_losses) > 1 else "")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Val Loss", f"{val_losses[-1]:.4f}",
              f"{val_losses[-1] - val_losses[-2]:.4f}" if len(val_losses) > 1 else "")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    best_epoch = np.argmin(val_losses) + 1
    st.metric("Best Epoch", best_epoch, f"Loss: {min(val_losses):.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Main visualization
tab1, tab2, tab3 = st.tabs(["🏔️ 3D Loss Terrain", "📈 Loss Curves", "🎯 Gradient Flow"])

with tab1:
    st.subheader("3D Loss Landscape")
    
    # Create meshgrid for 3D surface
    # We'll use PCA-like projection of parameter space to 2D, then plot loss
    # For simplicity, we'll use epochs and learning rate as x, y
    # This is a stylized visualization showing the optimization trajectory
    
    # Create a synthetic loss surface for visualization
    x = np.linspace(0, len(epochs), 50)
    y = np.linspace(min(learning_rates) * 0.5, max(learning_rates) * 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Interpolate actual losses onto the grid
    from scipy.interpolate import griddata
    points = np.array([[e, lr] for e, lr in zip(epochs, learning_rates)])
    Z_train = griddata(points, train_losses, (X, Y), method='cubic', fill_value=np.mean(train_losses))
    
    # Create 3D surface
    fig = go.Figure()
    
    # Add loss surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_train,
        colorscale='Viridis',
        opacity=0.9,
        name='Loss Surface',
        showscale=True,
        colorbar=dict(title="Loss")
    ))
    
    # Add optimization trajectory
    fig.add_trace(go.Scatter3d(
        x=epochs,
        y=learning_rates,
        z=train_losses,
        mode='lines+markers',
        marker=dict(
            size=3,
            color=epochs,
            colorscale='Plasma',
            showscale=False
        ),
        line=dict(color='white', width=2),
        name='Optimization Path'
    ))
    
    # Add starting point
    fig.add_trace(go.Scatter3d(
        x=[epochs[0]],
        y=[learning_rates[0]],
        z=[train_losses[0]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Start'
    ))
    
    # Add current point
    fig.add_trace(go.Scatter3d(
        x=[epochs[-1]],
        y=[learning_rates[-1]],
        z=[train_losses[-1]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name='Current'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Epoch',
            yaxis_title='Learning Rate',
            zaxis_title='Loss',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Loss Evolution")
    
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Loss Curves', 'Learning Rate Schedule'),
                        vertical_spacing=0.12)
    
    # Loss curves
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name='Train Loss',
                            line=dict(color='#00d4ff', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, name='Val Loss',
                            line=dict(color='#ff006e', width=2)), row=1, col=1)
    
    # Learning rate
    fig.add_trace(go.Scatter(x=epochs, y=learning_rates, name='LR',
                            fill='tozeroy',
                            line=dict(color='#8338ec', width=2)), row=2, col=1)
    
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1, type='log')
    fig.update_yaxes(title_text="Learning Rate", row=2, col=1, type='log')
    
    fig.update_layout(
        height=600,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Gradient Flow & Convergence")
    
    # Calculate gradient approximation (difference in loss)
    grad_approx = np.diff(train_losses)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs[1:],
        y=np.abs(grad_approx),
        mode='lines',
        fill='tozeroy',
        name='|Gradient|',
        line=dict(color='#06ffa5', width=2)
    ))
    
    fig.update_layout(
        title="Gradient Magnitude Over Time",
        xaxis_title="Epoch",
        yaxis_title="|∇Loss|",
        yaxis_type='log',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Convergence indicator
    recent_window = 50
    if len(grad_approx) > recent_window:
        recent_grad = np.mean(np.abs(grad_approx[-recent_window:]))
        if recent_grad < 0.001:
            st.success("🎉 Model appears to have converged!")
        elif recent_grad < 0.01:
            st.info("📊 Model is converging...")
        else:
            st.warning("⚠️ Model is still exploring the loss landscape")

# Footer
st.markdown("---")
st.markdown("**Tip**: Rotate the 3D plot by dragging, zoom with scroll wheel, and double-click to reset view.")
