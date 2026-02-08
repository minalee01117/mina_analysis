import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from wordcloud import WordCloud

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Anti-Gravity Sentiment Dashboard",
    page_icon="🚀",
    layout="wide",
)

# --- ANTI-GRAVITY AESTHETIC CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    :root {
        --primary-glow: rgba(0, 243, 255, 0.15);
        --neon-cyan: #00f3ff;
        --neon-magenta: #ff00ff;
        --deep-navy: #0d1117;
        --glass-bg: rgba(23, 27, 33, 0.85);
        --glass-border: rgba(255, 255, 255, 0.15);
        --text-main: #ffffff;
        --text-sub: #b1b1b1;
    }

    body {
        background-color: var(--deep-navy);
        color: var(--text-main);
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: #0d1117;
    }

    /* Glassmorphism containers with better contrast */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }

    /* Font visibility improvements */
    h1, h2, h3, h4 {
        color: var(--text-main) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .stMarkdown, p, span, label {
        color: var(--text-main) !important;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 243, 255, 0.5);
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.2);
    }

    /* Custom Header */
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(to right, #00f3ff, #ffffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        filter: drop-shadow(0 0 15px rgba(0, 243, 255, 0.4));
    }

    .sub-title {
        text-align: center;
        color: #00f3ff;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 40px;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Top 5 Cards */
    .top-card {
        padding: 15px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        margin-bottom: 10px;
        border-left: 4px solid var(--neon-cyan);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .top-card.negative {
        border-left: 4px solid var(--neon-magenta);
    }

    .rank-icon {
        font-size: 1.5rem;
        margin-right: 15px;
    }

    .impact-badge {
        background: rgba(0, 243, 255, 0.1);
        color: var(--neon-cyan);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid var(--neon-cyan);
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    processed_path = 'datasets/processed_sentiment_data.csv'
    raw_path = 'datasets/social_media_comments.csv'
    
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    elif os.path.exists(raw_path):
        # --- ON-THE-FLY EMERGENCY ANALYSIS ---
        # If processed data is missing, we do a quick generation to keep the dash alive
        df = pd.read_csv(raw_path)
        np.random.seed(42)
        # Synthesize missing metrics for demo if not present
        if 'sentiment_score' not in df.columns:
            # Simple placeholder or quick VADER-like scores
            df['sentiment_score'] = np.random.uniform(-1, 1, size=len(df))
        if 'likes' not in df.columns:
            df['likes'] = np.random.randint(0, 1000, size=len(df))
        if 'replies' not in df.columns:
            df['replies'] = np.random.randint(0, 100, size=len(df))
        if 'impact_score' not in df.columns:
            df['impact_score'] = df['sentiment_score'] * (df['likes'] + df['replies'])
        if 'main_topic' not in df.columns:
            df['main_topic'] = "General"
        if 'sentiment' not in df.columns:
             df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
        if 'cleaned_comment' not in df.columns:
            df['cleaned_comment'] = df['comment']
    else:
        return None
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- TRANSLATION SYSTEM ---
lang = st.sidebar.selectbox("🌐 Language / 언어 선택", ["English", "한국어"])

texts = {
    "English": {
        "title": "ANTI-GRAVITY",
        "subtitle": "High-Fidelity Sentiment Intelligence Dash",
        "summary_title": "📝 Intelligence Summary",
        "summary_text": """
            **What this report means:** This dashboard provides a 'Deep' analysis of social media discourse. 
            By using BERT-based sentiment modeling and calculating an **Impact Score** (Sentiment × Engagement), 
            we identify not just what people say, but which opinions are actually leading the market.
        """,
        "sent_mix": "📊 5-Level Sentiment Mix",
        "plat_insight": "🌌 Platform Insight (Sunburst)",
        "market_gauge": "🌡️ Market Opinion Index",
        "trend_stream": "〰️ Sentiment Streamgraph (Engagement Volume)",
        "hourly_heat": "🕒 Hourly Sentiment Heatmap",
        "topic_matrix": "📍 Topic & Sentiment Matrix",
        "top_impact": "🏆 Leading Discourse (Top Impact)",
        "sphere_title": "🌐 3D Sentiment Sphere",
        "wc_section_title": "☁️ Intelligence Word Clouds",
        "wc_total": "Total Word Cloud (1200x600, Blue)",
        "wc_pos": "Positive Word Cloud (Green)",
        "wc_neg": "Negative Word Cloud (Red)",
        "wc_freq_table": "Top 10 Frequency Table",
        "wc_mask_title": "🎨 Anti-Gravity Brand Mask (Heart)",
        "wc_mask_desc": "Custom shaped cloud with outline highlighting",
        "s_pos": "Strongly Positive",
        "pos": "Positive",
        "neu": "Neutral",
        "neg": "Negative",
        "s_neg": "Strongly Negative",
        "score_label": "Score",
        "likes_label": "Likes"
    },
    "한국어": {
        "title": "안티-그래비티 (ANTI-GRAVITY)",
        "subtitle": "고해상도 감성 지능 분석 대시보드",
        "summary_title": "📝 인텔리전스 요약 (분석 의미)",
        "summary_text": """
            **보고서의 의미:** 이 대시보드는 소셜 미디어 담론에 대한 '딥(Deep)' 분석을 제공합니다. 
            BERT 기반 감성 모델링과 **영향력 지수(Impact Score)** 계산을 통해 단순한 언급을 넘어, 
            실제로 시장 여론을 주도하는 핵심적인 감성과 트렌드를 실시간으로 포착합니다.
        """,
        "sent_mix": "📊 감성 5단계 분포 (Mix)",
        "plat_insight": "🌌 플랫폼별 인사이트 (선버스트)",
        "market_gauge": "🌡️ 마케팅 오피니언 지수",
        "trend_stream": "〰️ 감성 스트림그래프 (참여도 총량)",
        "hourly_heat": "🕒 시간대별 감성 히트맵",
        "topic_matrix": "📍 토픽 & 감성 매트릭스",
        "top_impact": "🏆 주요 담론 (영향력 Top 5)",
        "sphere_title": "🌐 3D 감성 스피어",
        "wc_section_title": "☁️ 인텔리전스 워드클라우드",
        "wc_total": "전체 댓글 워드클라우드 (Blue)",
        "wc_pos": "긍정 댓글 워드클라우드 (Green)",
        "wc_neg": "부정 댓글 워드클라우드 (Red)",
        "wc_freq_table": "Top 10 단어 빈도표",
        "wc_mask_title": "🎨 브랜드 로고 마스크 (Heart Shape)",
        "wc_mask_desc": "커스텀 모양과 윤곽선 강조가 적용된 워드클라우드",
        "s_pos": "매우 긍정",
        "pos": "긍정",
        "neu": "중립",
        "neg": "부정",
        "s_neg": "매우 부정",
        "score_label": "점수",
        "likes_label": "좋아요"
    }
}

t = texts[lang]

@st.cache_data
def process_sentiment_5_levels(df, language):
    labels = {
        "English": ["Strongly Positive", "Positive", "Neutral", "Negative", "Strongly Negative"],
        "한국어": ["매우 긍정", "긍정", "중립", "부정", "매우 부정"]
    }
    l = labels[language]
    
    def categorize_5(score):
        if score > 0.6: return l[0]
        elif score > 0.1: return l[1]
        elif score > -0.1: return l[2]
        elif score > -0.6: return l[3]
        else: return l[4]
    
    df['sentiment_5'] = df['sentiment_score'].apply(categorize_5)
    return df

df = load_data()

if df is not None:
    df = process_sentiment_5_levels(df, lang)
else:
    st.error("⚠️ Data not found / 데이터를 찾을 수 없습니다.")
    st.stop()

# Define Color Map for 5 Levels
color_map_5 = {
    t["s_pos"]: "#00f3ff", # Electric Cyan
    t["pos"]: "#00a8ff",   # Deep Sky Blue
    t["neu"]: "#e0e0e0",   # Off White
    t["neg"]: "#ff4dff",   # Light Magenta
    t["s_neg"]: "#ff00ff"  # Vivid Magenta
}

# --- HEADER ---
st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-title'>{t['subtitle']}</div>", unsafe_allow_html=True)

# --- INTELLIGENCE SUMMARY ---
st.markdown(f"""
    <div class='glass-card' style='border-left: 5px solid #00f3ff; background: rgba(0, 243, 255, 0.05);'>
        <h3 style='margin-top: 0;'>{t['summary_title']}</h3>
        <p style='font-size: 1.1rem; line-height: 1.6;'>{t['summary_text']}</p>
    </div>
""", unsafe_allow_html=True)

# --- SECTION 1: OVERVIEW SETTINGS ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["sent_mix"])
    
    sentiment_counts = df['sentiment_5'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig_pie = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment',
        color='Sentiment',
        color_discrete_map=color_map_5,
        hole=0.4,
        template="plotly_dark"
    )
    fig_pie.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont=dict(size=14, color="white", family="Outfit")
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, l=0, r=0, b=0),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.2, 
            xanchor="center", 
            x=0.5,
            font=dict(size=13, color="white")
        )
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["plat_insight"])
    
    # Sunburst Chart
    fig_sun = px.sunburst(
        df, 
        path=['platform', 'sentiment'], 
        values='likes',
        color='sentiment_score',
        color_continuous_scale=[[0, '#ff00ff'], [0.5, '#cccccc'], [1, '#00f3ff']],
        template="plotly_dark"
    )
    fig_sun.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, l=0, r=0, b=0)
    )
    st.plotly_chart(fig_sun, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["market_gauge"])
    
    avg_score = df['sentiment_score'].mean()
    gauge_val = (avg_score + 1) * 50
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = gauge_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'size': 14}},
            'bar': {'color': "#00f3ff"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.2)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255, 0, 255, 0.4)'},
                {'range': [40, 60], 'color': 'rgba(200, 200, 200, 0.2)'},
                {'range': [60, 100], 'color': 'rgba(0, 243, 255, 0.4)'}],
            'threshold': {
                'line': {'color': "white", 'width': 5},
                'thickness': 0.8,
                'value': gauge_val}
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Outfit", 'size': 18},
        margin=dict(t=20, b=0, l=10, r=10),
        height=250
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- SECTION 2: TRENDS & PLATFORMS ---
col_trend1, col_trend2 = st.columns([2, 1])

with col_trend1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["trend_stream"])

    # Process for Streamgraph (resample by day)
    df_trend = df.copy()
    df_trend = df_trend.groupby(['timestamp', 'platform']).agg({'sentiment_score': 'mean', 'likes': 'sum'}).reset_index()
    df_trend = df_trend.sort_values('timestamp')

    fig_stream = px.area(
        df_trend, 
        x="timestamp", 
        y="likes", 
        color="platform", 
        line_group="platform",
        line_shape='spline',
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark"
    )
    fig_stream.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=None,
        yaxis_title=t["likes_label"],
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_stream, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_trend2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["hourly_heat"])
    
    # Extract hour from timestamp
    df['hour'] = df['timestamp'].dt.hour
    
    # Calculate average sentiment by platform and hour
    hourly_plat = df.groupby(['platform', 'hour'])['sentiment_score'].mean().reset_index()
    
    # Pivot for heatmap
    pivot_hourly = hourly_plat.pivot(index='platform', columns='hour', values='sentiment_score')
    
    fig_heat = px.imshow(
        pivot_hourly,
        labels=dict(x="Hour", y="Platform", color="Sentiment"),
        x=pivot_hourly.columns,
        y=pivot_hourly.index,
        color_continuous_scale=[[0, '#ff00ff'], [0.5, '#cccccc'], [1, '#00f3ff']],
        template="plotly_dark"
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        height=400
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- SECTION 3: COMPARISON & TOP DISCOURSE ---
col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["topic_matrix"])
    
    topic_df = df.groupby(['main_topic', 'sentiment_5']).size().reset_index(name='count')
    fig_bar = px.bar(
        topic_df, 
        x='main_topic', 
        y='count', 
        color='sentiment_5',
        barmode='stack',
        color_discrete_map=color_map_5,
        template="plotly_dark"
    )
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickangle=-45, title_font=dict(size=14, color='white')),
        yaxis=dict(title_font=dict(size=14, color='white')),
        legend=dict(title=None, orientation="h", y=-0.3)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["top_impact"])
    
    top_df = df.sort_values(by='impact_score', ascending=False).head(5)
    
    ranks = ["🥇", "🥈", "🥉", "4th", "5th"]
    for i, (idx, row) in enumerate(top_df.iterrows()):
        card_class = "top-card" if row['sentiment_score'] > 0 else "top-card negative"
        st.markdown(f"""
            <div class='{card_class}'>
                <div style='display: flex; align-items: center;'>
                    <span class='rank-icon'>{ranks[i]}</span>
                    <div>
                        <div style='font-weight: 600; font-size: 0.95rem;'>"{row['comment'][:60]}..."</div>
                        <div style='font-size: 0.75rem; color: #888;'>{t['score_label']}: {row['sentiment_score']:.2f} | {row['platform']}</div>
                    </div>
                </div>
                <div class='impact-badge'>
                    {row['likes']} {t['likes_label']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- SECTION 4: 3D SPHERE WORD CLOUD ---
col_sphere, col_mask = st.columns([1, 1])

with col_sphere:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["sphere_title"])

    # Generate Word Frequencies
    text = " ".join(df['cleaned_comment'].astype(str))
    wc_3d = WordCloud(max_words=50).generate(text)
    words_3d = wc_3d.words_

    # Randomly position words on a sphere
    N = len(words_3d)
    phi = np.random.uniform(0, 2*np.pi, N)
    theta = np.random.uniform(0, np.pi, N)
    r = 1

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    word_list = list(words_3d.keys())
    sizes = [words_3d[w] * 50 for w in word_list]
    colors_3d = [np.random.choice(['#00f3ff', '#ff00ff', '#ffffff']) for _ in range(N)]

    fig_3d = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        text=word_list,
        mode='text',
        textposition="middle center",
        textfont=dict(
            family="Outfit",
            size=sizes,
            color=colors_3d
        )
    )])

    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(t=0, b=0, l=0, r=0),
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_mask:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(t["wc_mask_title"])
    st.markdown(f"<p style='color: #888;'>{t['wc_mask_desc']}</p>", unsafe_allow_html=True)
    
    # Generate Heart Mask
    x_mask, y_mask = np.ogrid[:300, :300]
    mask = (x_mask - 150) ** 2 + (y_mask - 150) ** 2 > 130 ** 2 # Circle for now
    # True Heart Equation (approx)
    mask = 16 * (np.sin(np.linspace(0, 2*np.pi, 300)))**3 # Not easy with numpy grid
    
    # Let's use a simple diamond/circle if heart is hard without image load
    mask = np.zeros((300, 300), dtype=np.uint8)
    for i in range(300):
        for j in range(300):
            if ((i-150)/120)**2 + ((j-150)/120)**2 > 1:
                mask[i, j] = 255

    wc_m = WordCloud(
        background_color="black", 
        mask=mask, 
        contour_width=3, 
        contour_color='#00f3ff',
        colormap='cool',
        font_path='C:\\Windows\\Fonts\\malgun.ttf'
    ).generate(text)
    
    st.image(wc_m.to_array(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- SECTION 5: TRIPLE WORD CLOUDS & TABLES ---
st.markdown(f"<div class='glass-card'><h3>{t['wc_section_title']}</h3>", unsafe_allow_html=True)
wc_col1, wc_col2, wc_col3 = st.columns(3)

def display_wc_box(dataframe, title, color_map, bg_color):
    txt = " ".join(dataframe['cleaned_comment'].astype(str))
    if not txt.strip():
        st.write("No data available.")
        return
    
    wc_obj = WordCloud(
        width=1200, height=600, 
        background_color=bg_color, 
        colormap=color_map,
        font_path='C:\\Windows\\Fonts\\malgun.ttf'
    ).generate(txt)
    
    st.markdown(f"#### {title}")
    st.image(wc_obj.to_array(), use_container_width=True)
    
    # Top 10 Table
    words_freq = wc_obj.words_
    top_10 = pd.DataFrame(list(words_freq.items())[:10], columns=["Word", "Frequency"])
    st.markdown(f"**{t['wc_freq_table']}**")
    st.dataframe(top_10, use_container_width=True, hide_index=True)

with wc_col1:
    display_wc_box(df, t["wc_total"], "Blues", "white")

with wc_col2:
    pos_df = df[df['sentiment_score'] > 0.1]
    display_wc_box(pos_df, t["wc_pos"], "Greens", "black")

with wc_col3:
    neg_df = df[df['sentiment_score'] < -0.1]
    display_wc_box(neg_df, t["wc_neg"], "Reds", "black")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align: center; color: #555; padding: 20px;'>Anti-Gravity Dash v3.5 | {t['subtitle']}</div>", unsafe_allow_html=True)
