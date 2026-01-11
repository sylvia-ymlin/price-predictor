"""
Generate architecture diagram for the Housing Prices Predictor System.
Uses matplotlib for visualization.
Updated to reflect Cloud Run, A/B Testing, and Observability.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    """Create the MLOps pipeline architecture diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Color scheme
    colors = {
        'data': '#BBDEFB',       # Light blue
        'preprocess': '#C8E6C9', # Light green
        'training': '#FFE0B2',   # Light orange
        'mlops': '#E1BEE7',      # Light purple
        'deployment': '#FFCDD2', # Light red
        'observability': '#B2DFDB', # Light teal
        'box': '#FFFFFF',        # White
        'border': '#455A64',     # Dark gray
        'text': '#212121',       # Near black
        'arrow': '#37474F',      # Blue gray
    }
    
    def draw_box(x, y, width, height, label, box_color='white'):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=box_color, edgecolor=colors['border'], linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color=colors['text'])
    
    def draw_section(x, y, width, height, title, bg_color):
        section = FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.02,rounding_size=0.15",
                                 facecolor=bg_color, edgecolor=colors['border'],
                                 linewidth=2, alpha=0.8)
        ax.add_patch(section)
        ax.text(x + width/2, y + height - 0.25, title, ha='center', va='top',
                fontsize=10, fontweight='bold', color=colors['text'])
    
    def draw_arrow(x1, y1, x2, y2, connection_style="arc3,rad=0"):
        # Manual arrow drawing using annotate for simple straight lines
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, connectionstyle=connection_style))

    # ==================== Row 1: Data + Preprocessing ====================
    
    # Data Layer
    draw_section(0.3, 8.8, 2.5, 1.8, "Data Layer", colors['data'])
    draw_box(0.5, 9.0, 0.9, 0.7, "Ames\nDataset", colors['box'])
    draw_box(1.7, 9.0, 0.9, 0.7, "Data\nIngestion", colors['box'])
    draw_arrow(1.4, 9.35, 1.7, 9.35)
    
    # Preprocessing
    draw_section(3.2, 8.8, 5.8, 1.8, "Preprocessing", colors['preprocess'])
    draw_box(3.4, 9.0, 1.1, 0.7, "Missing\nValues", colors['box'])
    draw_box(4.7, 9.0, 1.1, 0.7, "Feature\nEngineering", colors['box'])
    draw_box(6.0, 9.0, 1.1, 0.7, "Outlier\nDetection", colors['box'])
    draw_box(7.3, 9.0, 1.1, 0.7, "Train/Test\nSplit", colors['box'])
    
    draw_arrow(2.8, 9.35, 3.4, 9.35)  # Data -> Preprocessing
    draw_arrow(4.5, 9.35, 4.7, 9.35)
    draw_arrow(5.8, 9.35, 6.0, 9.35)
    draw_arrow(7.1, 9.35, 7.3, 9.35)
    
    # ==================== Row 2: Training + MLOps ====================
    
    # Model Training (Expanded)
    draw_section(0.3, 4.5, 6.0, 3.8, "Automated Training", colors['training'])
    draw_box(2.0, 7.2, 2.6, 0.6, "Hyperparameter Tuning\n(GridSearchCV)", colors['box'])
    
    draw_box(0.5, 6.0, 1.3, 0.6, "Linear Reg", colors['box'])
    draw_box(2.0, 6.0, 1.3, 0.6, "XGBoost", colors['box'])
    draw_box(3.5, 6.0, 1.3, 0.6, "Random Forest", colors['box'])
    
    draw_box(1.5, 5.0, 2.0, 0.6, "Model Evaluation\n(Evidently AI)", colors['box'])
    draw_box(3.8, 5.0, 1.8, 0.6, "Explainability\n(SHAP)", colors['box'])
    
    # Logic Arrows
    draw_arrow(7.85, 8.8, 7.85, 8.0)  # Down from Split
    draw_arrow(7.85, 8.0, 3.3, 8.0)   # Left
    draw_arrow(3.3, 8.0, 3.3, 7.8)    # Down to Tuning
    
    draw_arrow(3.3, 7.2, 1.15, 6.6)   # To Linear
    draw_arrow(3.3, 7.2, 2.65, 6.6)   # To XGBoost
    draw_arrow(3.3, 7.2, 4.15, 6.6)   # To RF
    
    draw_arrow(2.65, 6.0, 2.65, 5.6)  # XGBoost to Eval
    draw_arrow(2.65, 5.6, 4.7, 5.6)   # XGBoost to SHAP
    

    # MLOps Infrastructure
    draw_section(6.7, 4.5, 3.8, 3.8, "MLOps & Tracking", colors['mlops'])
    draw_box(7.0, 7.0, 1.5, 0.7, "MLflow\nTracking", colors['box'])
    draw_box(8.8, 7.0, 1.4, 0.7, "ZenML\nStore", colors['box'])
    draw_box(7.5, 5.5, 2.0, 0.7, "Model Registry\n(Production)", colors['box'])
    
    draw_arrow(3.5, 5.3, 7.0, 5.3)    # Eval -> Registry path (conceptual)
    draw_arrow(7.0, 5.3, 7.0, 7.0)    # Up to Tracking
    draw_arrow(7.75, 7.0, 7.75, 6.2)  # Connected
    
    # ==================== Row 3: Deployment ====================
    
    # Deployment Container
    draw_section(0.3, 0.5, 9.5, 3.5, "Cloud Deployment (GCP Cloud Run)", colors['deployment'])
    
    # Build System
    draw_box(0.5, 2.0, 2.0, 0.8, "Docker Build\n(Bundled Model)", colors['box'])
    
    # Cloud Run
    draw_box(3.0, 1.0, 3.5, 2.5, "", box_color='#FFEBEE') # Inner container
    ax.text(4.75, 3.3, "Traffic Splitting", ha='center', fontsize=9, fontweight='bold', color=colors['text'])
    
    draw_box(3.2, 1.2, 1.4, 0.8, "Rev 1 (Green)\n50% Traffic", "#A5D6A7") # Green
    draw_box(4.9, 1.2, 1.4, 0.8, "Rev 2 (Blue)\n50% Traffic", "#90CAF9")  # Blue
    
    draw_arrow(2.5, 2.4, 3.0, 2.4) # Build -> Cloud Run area
    
    # Load Balancer / User
    draw_box(3.0, 0.0, 3.5, 0.4, "Load Balancer / User Request", colors['box'])
    draw_arrow(4.75, 0.4, 4.75, 1.1)
    
    # Observability Stack
    draw_section(10.2, 0.5, 3.5, 3.5, "Observability", colors['observability'])
    draw_box(10.5, 2.5, 2.9, 0.6, "Prometheus\n(Metrics)", colors['box'])
    draw_box(10.5, 1.5, 2.9, 0.6, "Structured Logs\n(JSON)", colors['box'])
    draw_box(10.5, 0.8, 2.9, 0.5, "Grafana\n(Dashboard)", colors['box'])
    
    # Connect Deployment to Observability
    draw_arrow(6.4, 2.5, 10.2, 2.5) # Deployment -> Monitoring
    
    # ==================== Title ====================
    ax.text(7, 11.2, "Housing Prices Predictor - Advanced MLOps Architecture",
            ha='center', va='center', fontsize=16, fontweight='bold', color=colors['text'])
            
    # Add Legend/Notes
    ax.text(12, 11.2, "v2.0", ha='right', fontsize=12, color='gray')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = create_architecture_diagram()
    
    # Save as PNG
    output_path = "./docs/architecture_diagram.png"
    import os
    os.makedirs("./docs", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to: {output_path}")
