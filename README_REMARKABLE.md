# Anti-Gravity Sentiment Intelligence Analysis 🚀

This repository contains an advanced sentiment analysis engine and a premium interactive dashboard designed for deep social media discourse analysis.

## ✨ Key Features

- **🧠 Deep Sentiment Engine**: Uses BERT-based transformer models for context-aware sentiment analysis (DistilBERT/KcBERT).
- **📈 Impact Score**: Calculates influence by combining sentiment with user engagement (likes + replies).
- **🎨 Anti-Gravity Aesthetic**: Premium Streamlit dashboard featuring glassmorphism, neon glow, and futuristic UI.
- **🕒 Multidimensional Visualization**:
    - **Sunburst Chart**: Hierarchical platform-sentiment overview.
    - **Hourly Heatmap**: Temporal sentiment trends.
    - **3D Sentiment Sphere**: Interactive 3D word cloud.
    - **Custom Masked Word Cloud**: Brand-shaped word clouds with contour highlighting.
- **🌐 Bilingual Support**: Seamlessly switch between English and Korean.

## 🛠️ Project Structure

- `antigravity_analysis.py`: The core NLP engine that processes raw CSV data and calculates deep sentiment scores.
- `antigravity_dashboard.py`: The high-fidelity Streamlit dashboard application.
- `datasets/`: Directory containing raw and processed CSV data.
- `requirements.txt`: Python dependencies.

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Analysis Engine

Process your raw data through the BERT sentiment model:

```bash
python antigravity_analysis.py
```

### 3. Launch Dashboard

Start the premium interactive dashboard:

```bash
streamlit run antigravity_dashboard.py
```

## 📊 Analytics Methodology

Unlike standard VADER analysis, this project leverages **Transfer Learning** to understand nuance and sarcasm. The **Impact Score** ensures that highly engaged opinions (viral comments) are given more weight in the final report, providing a true reflection of market sentiment.

---
*Developed for Advanced Agentic Coding - Anti-Gravity v3.5*
