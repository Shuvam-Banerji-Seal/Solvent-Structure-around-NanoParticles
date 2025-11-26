import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

# Page config
st.set_page_config(page_title="Latent Space Explorer", layout="wide")

st.title("🌌 Latent Space Explorer")
st.markdown("Visualize the 512-dimensional latent space in 3D to understand how the model encodes solvent effects.")

# Load Data
@st.cache_data
def load_data():
    # Path relative to script location
    script_dir = Path(__file__).parent
    path = script_dir.parent / "logs" / "latent_vectors.csv"
    if not path.exists():
        st.error(f"File not found: {path.absolute()}")
        return None
    return pd.read_csv(path)

df = load_data()

if df is not None:
    # Sidebar Controls
    st.sidebar.header("Controls")
    
    # Epoch Selection
    epochs = sorted(df['epoch'].unique())
    selected_epoch = st.sidebar.select_slider("Select Epoch", options=epochs, value=epochs[-1])
    
    # Filter data
    epoch_data = df[df['epoch'] == selected_epoch].copy()
    
    # Dimensionality Reduction Method
    method = st.sidebar.radio("Projection Method", ["PCA", "t-SNE"])
    
    # Extract latent vectors (z_0 to z_511)
    z_cols = [c for c in df.columns if c.startswith('z_')]
    X = epoch_data[z_cols].values
    
    # Compute Projection
    @st.cache_data
    def compute_projection(X, method, perplexity=30):
        if method == "PCA":
            pca = PCA(n_components=3)
            return pca.fit_transform(X)
        else:
            tsne = TSNE(n_components=3, perplexity=min(perplexity, len(X)-1), random_state=42)
            return tsne.fit_transform(X)
            
    perplexity = 30
    if method == "t-SNE":
        perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
        
    with st.spinner(f"Computing {method}..."):
        if len(X) < 3:
            st.warning(f"Not enough data points for {method} (need at least 3, got {len(X)}).")
            # Create dummy projection for visualization to not crash
            proj = np.zeros((len(X), 3))
            if len(X) > 0:
                proj[:, 0] = np.arange(len(X))
        else:
            proj = compute_projection(X, method, perplexity)
        
    epoch_data['x'] = proj[:, 0]
    epoch_data['y'] = proj[:, 1]
    epoch_data['z'] = proj[:, 2]
    
    # Main Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"3D Latent Terrain (Epoch {selected_epoch})")
        
        # 3D Scatter
        fig = px.scatter_3d(
            epoch_data, x='x', y='y', z='z',
            color='epsilon',
            color_continuous_scale='Viridis',
            hover_data=['epsilon'],
            title=f"{method} Projection of Latent Space"
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Manifold Analysis")
        st.write(f"**Number of Points:** {len(epoch_data)}")
        st.write(f"**Epsilon Range:** {epoch_data['epsilon'].min():.2f} - {epoch_data['epsilon'].max():.2f}")
        
        # Surface Plot (Terrain)
        st.markdown("### 🏔️ Terrain View")
        st.markdown("Interpolated surface showing the smooth manifold structure.")
        
        # Create mesh
        fig_surf = go.Figure(data=[go.Mesh3d(
            x=epoch_data['x'],
            y=epoch_data['y'],
            z=epoch_data['z'],
            intensity=epoch_data['epsilon'],
            colorscale='Viridis',
            opacity=0.8
        )])
        fig_surf.update_layout(
            title="Latent Manifold Surface",
            scene=dict(
                xaxis_title='Dim 1',
                yaxis_title='Dim 2',
                zaxis_title='Dim 3'
            ),
            height=400,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig_surf, use_container_width=True)
        
    # Trajectory Analysis (All Epochs)
    st.markdown("---")
    st.subheader("📈 Learning Trajectory")
    st.markdown("How a specific epsilon's representation evolves over training.")
    
    target_eps = st.selectbox("Select Epsilon to Track", sorted(df['epsilon'].unique()))
    
    track_data = df[df['epsilon'] == target_eps].copy()
    
    # We need to project ALL data to consistent space to see trajectory?
    # Or just track statistics?
    # Projecting all epochs is expensive.
    # Let's plot the norm of the latent vector over time.
    
    track_data['norm'] = np.linalg.norm(track_data[z_cols].values, axis=1)
    
    fig_traj = px.line(track_data, x='epoch', y='norm', title=f"Latent Vector Norm Evolution (Epsilon {target_eps})")
    st.plotly_chart(fig_traj, use_container_width=True)

else:
    st.warning("No data loaded. Please ensure training has run and logs exist.")
