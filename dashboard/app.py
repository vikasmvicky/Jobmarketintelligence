import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import warnings
import os

warnings.filterwarnings('ignore')

# ── Page Config ──
st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Path Calculation (Absolute Paths) ──
# __file__ points to /opt/render/project/src/dashboard/app.py
# We go up two levels to get to /opt/render/project/src/
# Then we add /data or /models
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    # Fallback for local execution if __file__ isn't defined (rare)
    BASE_DIR = os.getcwd()

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #185FA5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .big-number {
        font-size: 2rem;
        font-weight: 700;
        color: #185FA5;
    }
    .insight-box {
        background: #E6F1FB;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #378ADD;
        margin: 0.5rem 0;
    }
    h1 { color: #1F4E79 !important; }
    h2 { color: #185FA5 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Data (Fixed with Absolute Paths) ──
@st.cache_data
def load_data():
    # Construct absolute path
    data_path = os.path.join(BASE_DIR, 'data', 'data_roles_final.csv')
    
    # DEBUG: These will appear in Render Logs
    print(f"DEBUG: Script Dir: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"DEBUG: Base Dir: {BASE_DIR}")
    print(f"DEBUG: Data Path: {data_path}")
    print(f"DEBUG: File Exists? {os.path.exists(data_path)}")

    df = pd.read_csv(data_path)
    
    df_sal = df[
        df['salary_final'].notna() &
        (df['salary_final'] > 20000) &
        (df['salary_final'] < 400000)
    ].copy()
    
    return df, df_sal

# ── Load Model (Fixed with Absolute Paths) ──
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
    label_map_path = os.path.join(BASE_DIR, 'models', 'label_map.json')
    
    print(f"DEBUG: Model Path: {model_path}")
    
    model = joblib.load(model_path)
    
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    
    label_map = {int(k): v for k, v in label_map.items()}
    return model, label_map

# ── Initialize ──
df, df_sal = load_data()
model, label_map = load_model()

# ── Sidebar ──
st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart.png",
    width=80)
st.sidebar.title("📊 Job Market Intel")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📊 Market Analysis",
     "🔮 Salary Predictor",
     "💡 Skill Gap Finder"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dataset Info**
- 📁 8,819 Data Roles
- 🌍 US Job Market
- 📅 April 2024
- 🤖 Gradient Boosting Model
- 🎯 Accuracy: 67.78%
""")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by **Vikas M Vicky** 🚀")

# ══════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════
if page == "🏠 Overview":
    st.title("📊 Global Data Job Market Intelligence")
    st.markdown(
        "##### Analyzing 8,819 data roles — "
        "Skills, Salaries & Career Insights")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='color:#888;font-size:13px'>
            Total Roles</div>
            <div class='big-number'>8,819</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='color:#888;font-size:13px'>
            Entry Level Salary</div>
            <div class='big-number'>$84K</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='color:#888;font-size:13px'>
            Full-Time Jobs</div>
            <div class='big-number'>77%</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div style='color:#888;font-size:13px'>
            Remote Allowed</div>
            <div class='big-number'>14.4%</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class='metric-card'>
            <div style='color:#888;font-size:13px'>
            Model Accuracy</div>
            <div class='big-number'>67.8%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Key Insights
    st.subheader("🔥 Top Findings From The Data")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class='insight-box'>
        <b>💡 Insight 1 — Skills beat Experience</b><br>
        Python + ML skills impact salary MORE than
        years of experience. Technical skills are
        the #1 salary driver in 2026.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
        <b>💡 Insight 2 — The 52% Opportunity</b><br>
        Adding Python, ML, AWS & Spark increases
        high salary probability from 36% → 89%.
        A 52% improvement from just 4 skills.
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class='insight-box'>
        <b>💡 Insight 3 — Business Analyst > Data Scientist</b><br>
        Business Analyst is more in demand (146 postings)
        than Data Scientist (56 postings). Business
        thinking is valued over pure ML in 2026.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
        <b>💡 Insight 4 — First Job = Biggest Salary Jump</b><br>
        Internship pays $56K vs Entry Level $84K.
        A 50% salary increase happens at your
        very first full-time job.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Salary ladder chart
    st.subheader("📈 Salary Ladder by Experience")
    exp_data = pd.DataFrame({
        'Experience': ['Internship', 'Entry Level',
                       'Associate', 'Mid-Senior',
                       'Director', 'Executive'],
        'Salary': [56160, 84318, 82330,
                   114550, 212000, 215000]
    })

    fig = px.bar(exp_data,
                 x='Experience', y='Salary',
                 color='Salary',
                 color_continuous_scale='Blues',
                 text='Salary')
    fig.update_traces(
        texttemplate='$%{text:,.0f}',
        textposition='outside')
    fig.update_layout(
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════
# PAGE 2 — MARKET ANALYSIS
# ══════════════════════════════════════
elif page == "📊 Market Analysis":
    st.title("📊 Market Analysis")
    st.markdown("Deep dive into job market trends")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["🏆 Top Roles", "🌆 Top Cities", "💼 Work Type"])

    with tab1:
        st.subheader("Most In-Demand Data Roles")
        title_counts = df['title'].value_counts().head(15)
        fig = px.bar(
            x=title_counts.values,
            y=title_counts.index,
            orientation='h',
            color=title_counts.values,
            color_continuous_scale='Blues',
            labels={'x': 'Job Postings', 'y': 'Job Title'})
        fig.update_layout(
            height=500,
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Top Cities for Data Jobs")
        city_counts = df['city'].value_counts().head(15)
        fig = px.bar(
            x=city_counts.index,
            y=city_counts.values,
            color=city_counts.values,
            color_continuous_scale='Blues',
            labels={'x': 'City', 'y': 'Job Postings'})
        fig.update_layout(
            height=450,
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Work Type Breakdown")
        work_counts = df['formatted_work_type'].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                values=work_counts.values,
                names=work_counts.index,
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Key Numbers")
            for wtype, count in work_counts.items():
                pct = count / len(df) * 100
                st.metric(wtype, f"{count:,}",
                          f"{pct:.1f}% of market")

# ══════════════════════════════════════
# PAGE 3 — SALARY PREDICTOR
# ══════════════════════════════════════
elif page == "🔮 Salary Predictor":
    st.title("🔮 Salary Bracket Predictor")
    st.markdown(
        "Enter your profile to predict your salary bracket")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👤 Your Profile")

        exp_level = st.selectbox(
            "Experience Level",
            ['Internship', 'Entry level', 'Associate',
             'Mid-Senior level', 'Director', 'Executive'])

        job_title = st.selectbox(
            "Job Title Category",
            ['data_analyst', 'data_scientist',
             'data_engineer', 'business_analyst',
             'financial_analyst', 'ml_engineer',
             'senior_role', 'manager', 'director', 'other'])

        is_remote = st.checkbox("Remote Job?")
        is_fulltime = st.checkbox("Full-Time?", value=True)

        st.subheader("🛠 Your Skills")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            has_python = st.checkbox("Python")
            has_sql = st.checkbox("SQL")
            has_excel = st.checkbox("Excel")
            has_tableau = st.checkbox("Tableau")
            has_powerbi = st.checkbox("Power BI")
        with col_s2:
            has_ml = st.checkbox("Machine Learning")
            has_aws = st.checkbox("AWS")
            has_spark = st.checkbox("Spark")
            has_r = st.checkbox("R Programming")
            has_stats = st.checkbox("Statistics")

    with col2:
        st.subheader("🎯 Prediction Result")

        exp_map = {
            'Internship': 0, 'Entry level': 1,
            'Associate': 2, 'Mid-Senior level': 3,
            'Director': 4, 'Executive': 5
        }
        title_map = {
            'business_analyst': 0, 'data_analyst': 1,
            'data_engineer': 2, 'data_scientist': 3,
            'director': 4, 'financial_analyst': 5,
            'manager': 6, 'ml_engineer': 7,
            'other': 8, 'senior_role': 9
        }

        features = pd.DataFrame([{
            'is_remote': int(is_remote),
            'is_fulltime': int(is_fulltime),
            'exp_encoded': exp_map[exp_level],
            'title_encoded': title_map[job_title],
            'has_skills': 1,
            'views': 100, 'applies': 10,
            'apply_rate': 0.1,
            'skill_python': int(has_python),
            'skill_sql': int(has_sql),
            'skill_excel': int(has_excel),
            'skill_tableau': int(has_tableau),
            'skill_power_bi': int(has_powerbi),
            'skill_machine_learning': int(has_ml),
            'skill_aws': int(has_aws),
            'skill_spark': int(has_spark),
            'skill_r_programming': int(has_r),
            'skill_statistics': int(has_stats)
        }])

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        # Result display
        colors_map = {
            0: '🔴', 1: '🟡', 2: '🟢'}
        st.markdown(f"""
        <div style='background:white;padding:2rem;
        border-radius:12px;text-align:center;
        box-shadow:0 4px 12px rgba(0,0,0,0.1);
        border-top:4px solid #185FA5;'>
        <div style='font-size:3rem'>
        {colors_map[pred]}</div>
        <div style='font-size:1.8rem;font-weight:700;
        color:#1F4E79;margin:0.5rem 0'>
        {label_map[pred]}</div>
        <div style='color:#555;font-size:14px'>
        Confidence: {max(prob)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Probability Breakdown")
        salary_levels = ['Low (<$60k)',
                         'Mid ($60k-$120k)',
                         'High (>$120k)']
        fig = go.Figure(go.Bar(
            x=prob * 100,
            y=salary_levels,
            orientation='h',
            marker_color=['#E24B4A', '#EF9F27', '#639922'],
            text=[f'{p*100:.1f}%' for p in prob],
            textposition='outside'))
        fig.update_layout(
            height=250,
            xaxis_title='Probability (%)',
            xaxis_range=[0, 110],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════
# PAGE 4 — SKILL GAP FINDER
# ══════════════════════════════════════
elif page == "💡 Skill Gap Finder":
    st.title("💡 Skill Gap Finder")
    st.markdown(
        "Discover which skills will boost "
        "your salary the most")
    st.markdown("---")

    st.subheader("✅ Select Skills You Already Have")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        s_python = st.checkbox("🐍 Python")
    with col2:
        s_sql = st.checkbox("🗄️ SQL")
    with col3:
        s_excel = st.checkbox("📊 Excel")
    with col4:
        s_ml = st.checkbox("🤖 ML")
    with col5:
        s_aws = st.checkbox("☁️ AWS")

    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        s_tableau = st.checkbox("📈 Tableau")
    with col7:
        s_powerbi = st.checkbox("📉 Power BI")
    with col8:
        s_spark = st.checkbox("⚡ Spark")
    with col9:
        s_r = st.checkbox("📐 R")
    with col10:
        s_stats = st.checkbox("📏 Statistics")

    if st.button("🔍 Find My Skill Gaps",
                 use_container_width=True):

        # Base profile with current skills
        base = {
            'is_remote': 0, 'is_fulltime': 1,
            'exp_encoded': 1, 'title_encoded': 2,
            'has_skills': 1, 'views': 100,
            'applies': 10, 'apply_rate': 0.1,
            'skill_python': int(s_python),
            'skill_sql': int(s_sql),
            'skill_excel': int(s_excel),
            'skill_tableau': int(s_tableau),
            'skill_power_bi': int(s_powerbi),
            'skill_machine_learning': int(s_ml),
            'skill_aws': int(s_aws),
            'skill_spark': int(s_spark),
            'skill_r_programming': int(s_r),
            'skill_statistics': int(s_stats)
        }

        base_prob = model.predict_proba(
            pd.DataFrame([base]))[0][2] * 100

        st.markdown("---")
        st.subheader("📊 Your Current Profile")
        st.metric("High Salary Probability",
                  f"{base_prob:.1f}%")

        # Test adding each missing skill
        missing_skills = {
            'skill_python': ('Python', s_python),
            'skill_machine_learning': ('Machine Learning', s_ml),
            'skill_aws': ('AWS', s_aws),
            'skill_spark': ('Spark', s_spark),
            'skill_sql': ('SQL', s_sql),
            'skill_statistics': ('Statistics', s_stats),
            'skill_tableau': ('Tableau', s_tableau),
            'skill_power_bi': ('Power BI', s_powerbi),
        }

        skill_gains = []
        for skill_col, (skill_name, has_it) in \
                missing_skills.items():
            if not has_it:
                test = base.copy()
                test[skill_col] = 1
                new_prob = model.predict_proba(
                    pd.DataFrame([test]))[0][2] * 100
                gain = new_prob - base_prob
                if gain > 0:
                    skill_gains.append({
                        'Skill': skill_name,
                        'Current Prob': base_prob,
                        'New Prob': new_prob,
                        'Gain': gain
                    })

        if skill_gains:
            skill_gains_df = pd.DataFrame(
                skill_gains).sort_values(
                'Gain', ascending=False)

            st.subheader(
                "🎯 Skills That Will Boost Your Salary Most")

            for _, row in skill_gains_df.iterrows():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f"**{row['Skill']}**")
                    st.progress(
                        min(row['New Prob'] / 100, 1.0))
                with col_b:
                    st.metric(
                        "Salary Boost",
                        f"+{row['Gain']:.1f}%")

            st.markdown("---")
            st.markdown("""
            <div class='insight-box'>
            <b>💡 Key Finding:</b> Adding Python, ML, 
            AWS and Spark increases high salary 
            probability from 36% → 89% (+52.4%).
            Skills matter MORE than experience!
            </div>""", unsafe_allow_html=True)
        else:
            st.success(
                "🎉 You already have all key skills! "
                "Focus on experience level now.")