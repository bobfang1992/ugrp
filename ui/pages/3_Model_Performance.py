"""
Model Performance page - visualize evaluation metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

st.set_page_config(
    page_title="UGRP - Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance")
st.markdown("Evaluation metrics on held-out test data (temporal split, 20% per user)")

# Load evaluation results
@st.cache_data
def load_evaluation(dataset="ml-1m"):
    """Load evaluation results for a dataset"""
    data_dir = Path("data/processed")

    if dataset == "ml-20m":
        eval_file = data_dir / "evaluation_20m.json"
    else:
        eval_file = data_dir / "evaluation.json"

    if not eval_file.exists():
        return None

    with open(eval_file, 'r') as f:
        results = json.load(f)

    return results


# Dataset selector
st.sidebar.header("⚙️ Settings")
dataset = st.sidebar.radio(
    "Dataset:",
    options=["ml-1m", "ml-20m"],
    format_func=lambda x: "MovieLens 1M (6K users)" if x == "ml-1m" else "MovieLens 20M (138K users)"
)

# Load evaluation
eval_results = load_evaluation(dataset)

if eval_results is None:
    st.error(f"❌ Evaluation results not found for {dataset.upper()}")
    st.info("""
    Train the model with evaluation to generate results:
    ```bash
    # Process data with train/test split
    python src/ugrp/recsys/data_loader.py --dataset {dataset}

    # Train and evaluate
    python src/ugrp/recsys/model.py --dataset {dataset}
    ```
    """)
    st.stop()

# Display key metrics
st.header(f"Evaluation Results: {dataset.upper()}")
st.caption(f"Evaluated on {eval_results['num_evaluated_users']:,} test users")

# Extract K values and metrics
k_values = sorted(set(int(key.split('@')[1]) for key in eval_results.keys() if '@' in key))

# Metrics overview
st.subheader("📈 Metrics Overview")

cols = st.columns(len(k_values))
for i, k in enumerate(k_values):
    with cols[i]:
        st.metric(f"NDCG@{k}", f"{eval_results[f'NDCG@{k}']:.4f}")
        st.metric(f"Precision@{k}", f"{eval_results[f'P@{k}']:.4f}")
        st.metric(f"Recall@{k}", f"{eval_results[f'R@{k}']:.4f}")
        st.metric(f"Hit Rate@{k}", f"{eval_results[f'HR@{k}']:.4f}")

# Detailed metrics table
st.subheader("📋 Detailed Metrics")

metrics_data = []
for k in k_values:
    metrics_data.append({
        'K': k,
        'Precision@K': eval_results[f'P@{k}'],
        'Recall@K': eval_results[f'R@{k}'],
        'NDCG@K': eval_results[f'NDCG@{k}'],
        'Hit Rate@K': eval_results[f'HR@{k}'],
        'MAP@K': eval_results[f'MAP@{k}']
    })

metrics_df = pd.DataFrame(metrics_data)

st.dataframe(
    metrics_df.style.format({
        'Precision@K': '{:.4f}',
        'Recall@K': '{:.4f}',
        'NDCG@K': '{:.4f}',
        'Hit Rate@K': '{:.4f}',
        'MAP@K': '{:.4f}'
    }),
    use_container_width=True,
    hide_index=True
)

# Visualizations
st.subheader("📊 Metric Comparisons")

col1, col2 = st.columns(2)

with col1:
    # Precision, Recall, NDCG comparison
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Precision@K',
        x=[f'K={k}' for k in k_values],
        y=[eval_results[f'P@{k}'] for k in k_values],
        marker_color='#636EFA'
    ))

    fig.add_trace(go.Bar(
        name='Recall@K',
        x=[f'K={k}' for k in k_values],
        y=[eval_results[f'R@{k}'] for k in k_values],
        marker_color='#EF553B'
    ))

    fig.add_trace(go.Bar(
        name='NDCG@K',
        x=[f'K={k}' for k in k_values],
        y=[eval_results[f'NDCG@{k}'] for k in k_values],
        marker_color='#00CC96'
    ))

    fig.update_layout(
        title='Precision, Recall, NDCG Comparison',
        barmode='group',
        xaxis_title='K',
        yaxis_title='Score',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Hit Rate and MAP
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Hit Rate@K',
        x=[f'K={k}' for k in k_values],
        y=[eval_results[f'HR@{k}'] for k in k_values],
        marker_color='#AB63FA'
    ))

    fig.add_trace(go.Bar(
        name='MAP@K',
        x=[f'K={k}' for k in k_values],
        y=[eval_results[f'MAP@{k}'] for k in k_values],
        marker_color='#FFA15A'
    ))

    fig.update_layout(
        title='Hit Rate and MAP Comparison',
        barmode='group',
        xaxis_title='K',
        yaxis_title='Score',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# Line chart showing metric trends across K
st.subheader("📈 Metric Trends Across K")

fig = go.Figure()

metrics_to_plot = ['NDCG', 'P', 'R', 'HR', 'MAP']
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

for metric, color in zip(metrics_to_plot, colors):
    fig.add_trace(go.Scatter(
        name=f'{metric}@K',
        x=k_values,
        y=[eval_results[f'{metric}@{k}'] for k in k_values],
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=8)
    ))

fig.update_layout(
    xaxis_title='K',
    yaxis_title='Score',
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Comparison section if both datasets exist
st.markdown("---")
st.subheader("🔄 Dataset Comparison")

# Try to load both datasets
ml1m_results = load_evaluation("ml-1m")
ml20m_results = load_evaluation("ml-20m")

if ml1m_results and ml20m_results:
    st.success("✅ Both datasets available for comparison")

    # Create comparison table
    comparison_data = []
    for k in k_values:
        comparison_data.append({
            'Metric': f'NDCG@{k}',
            'ML-1M': ml1m_results[f'NDCG@{k}'],
            'ML-20M': ml20m_results[f'NDCG@{k}'],
            'Difference': ml20m_results[f'NDCG@{k}'] - ml1m_results[f'NDCG@{k}']
        })
        comparison_data.append({
            'Metric': f'Precision@{k}',
            'ML-1M': ml1m_results[f'P@{k}'],
            'ML-20M': ml20m_results[f'P@{k}'],
            'Difference': ml20m_results[f'P@{k}'] - ml1m_results[f'P@{k}']
        })
        comparison_data.append({
            'Metric': f'Recall@{k}',
            'ML-1M': ml1m_results[f'R@{k}'],
            'ML-20M': ml20m_results[f'R@{k}'],
            'Difference': ml20m_results[f'R@{k}'] - ml1m_results[f'R@{k}']
        })
        comparison_data.append({
            'Metric': f'Hit Rate@{k}',
            'ML-1M': ml1m_results[f'HR@{k}'],
            'ML-20M': ml20m_results[f'HR@{k}'],
            'Difference': ml20m_results[f'HR@{k}'] - ml1m_results[f'HR@{k}']
        })

    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(
        comparison_df.style.format({
            'ML-1M': '{:.4f}',
            'ML-20M': '{:.4f}',
            'Difference': '{:+.4f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    # Comparison chart
    fig = go.Figure()

    for k in k_values:
        fig.add_trace(go.Bar(
            name=f'ML-1M (NDCG@{k})',
            x=[f'NDCG@{k}'],
            y=[ml1m_results[f'NDCG@{k}']],
            marker_color='lightblue',
            showlegend=False
        ))

        fig.add_trace(go.Bar(
            name=f'ML-20M (NDCG@{k})',
            x=[f'NDCG@{k}'],
            y=[ml20m_results[f'NDCG@{k}']],
            marker_color='darkblue',
            showlegend=False
        ))

    fig.update_layout(
        title='NDCG Comparison: ML-1M vs ML-20M',
        xaxis_title='Metric',
        yaxis_title='Score',
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    missing = []
    if not ml1m_results:
        missing.append("ML-1M")
    if not ml20m_results:
        missing.append("ML-20M")

    st.info(f"📝 Missing evaluation results for: {', '.join(missing)}")
    st.markdown("Train both models to enable comparison.")

# Footer - Methodology
st.markdown("---")
with st.expander("📖 Evaluation Methodology"):
    st.markdown("""
    ### Train/Test Split

    - **Method**: Temporal split per user
    - **Test Size**: 20% of each user's ratings (most recent by timestamp)
    - **Train Size**: 80% of each user's ratings (earliest by timestamp)
    - **Minimum Train Items**: 5 ratings required to include user in test set

    ### Metrics Explained

    - **Precision@K**: Fraction of recommended items (in top-K) that are relevant
    - **Recall@K**: Fraction of relevant items that appear in top-K recommendations
    - **NDCG@K**: Normalized Discounted Cumulative Gain - measures ranking quality with position weighting
    - **Hit Rate@K**: Percentage of users with at least one relevant item in top-K
    - **MAP@K**: Mean Average Precision - average precision at each relevant item position

    ### Relevance Definition

    A test item is considered "relevant" if the user rated it in the test set (held-out future ratings).
    """)
