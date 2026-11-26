"""
Student Performance Prediction System
======================================
A modern academic analytics dashboard built with Streamlit.
Predicts students' final examination marks (G3) using Linear Regression
trained on the UCI Student Performance (Mathematics) dataset.

Run:  streamlit run app.py
"""

# ── Standard library ────────────────────────────────────────────────────
import os
import datetime

# ── Third-party ─────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project modules ─────────────────────────────────────────────────────
from model.train_model import load_data, train_model, predict_grade

# =====================================================================
# Page config & custom CSS
# =====================================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.isfile(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =====================================================================
# Plotly dark-theme defaults
# =====================================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,27,45,0.6)",
    font=dict(family="Inter, sans-serif", color="#e8eaf6"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(74,144,217,0.1)", zerolinecolor="rgba(74,144,217,0.15)"),
    yaxis=dict(gridcolor="rgba(74,144,217,0.1)", zerolinecolor="rgba(74,144,217,0.15)"),
    hoverlabel=dict(bgcolor="#162236", font_color="#e8eaf6", bordercolor="#4a90d9"),
    colorway=["#4a90d9", "#7c4dff", "#64b5f6", "#b388ff", "#66bb6a",
              "#ffa726", "#ef5350", "#26c6da", "#ab47bc", "#8d6e63"],
)

# =====================================================================
# Session-state defaults
# =====================================================================
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# =====================================================================
# Cached data & model loading
# =====================================================================
@st.cache_data(show_spinner=False)
def get_data():
    return load_data()

@st.cache_resource(show_spinner=False)
def get_model(_df):
    return train_model(_df)

# ── Attempt to load data & train model ──────────────────────────────
DATA_LOADED = True
try:
    df = get_data()
    result = get_model(df)
except FileNotFoundError as exc:
    DATA_LOADED = False
    data_error = str(exc)
except Exception as exc:
    DATA_LOADED = False
    data_error = f"An unexpected error occurred: {exc}"

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🎓</div>
            <div class="sidebar-logo-text">EduPredict</div>
            <div class="sidebar-logo-sub">Performance Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "📊  Dashboard",
            "🎯  Predict Performance",
            "📈  Student Analytics",
            "🗃️  Dataset Explorer",
            "🤖  Model Information",
            "ℹ️  About Project",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("© 2026 EduPredict · v1.0")

# ── Show error splash if data could not be loaded ───────────────────
if not DATA_LOADED:
    st.error(data_error)
    st.info(
        "**How to fix:**\n\n"
        "1. Download the dataset from the [UCI ML Repository]"
        "(https://archive.ics.uci.edu/dataset/320/student+performance).\n"
        "2. Extract **student-mat.csv** and place it in the `data/` folder "
        "of this project.\n"
        "3. Restart the app."
    )
    st.stop()

# Convenience references
model = result["model"]
mae = result["mae"]
r2 = result["r2"]
y_test = result["y_test"]
y_pred = result["y_pred"]

# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

def grade_label(score: float) -> tuple[str, str]:
    """Return (label, css_class) for a predicted score."""
    if score >= 16:
        return "Excellent", "grade-excellent"
    elif score >= 12:
        return "Good", "grade-good"
    elif score >= 8:
        return "Average", "grade-average"
    else:
        return "Needs Improvement", "grade-poor"

def build_gauge(score: float):
    """Build a Plotly gauge chart for the predicted score."""
    label, _ = grade_label(score)
    color_map = {"Excellent": "#66bb6a", "Good": "#64b5f6",
                 "Average": "#ffa726", "Needs Improvement": "#ef5350"}
    bar_color = color_map.get(label, "#4a90d9")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix=" / 20", font=dict(size=36, color="#e8eaf6")),
        gauge=dict(
            axis=dict(range=[0, 20], tickcolor="#90a4ae",
                      tickfont=dict(color="#90a4ae")),
            bar=dict(color=bar_color, thickness=0.35),
            bgcolor="rgba(30,48,80,0.5)",
            borderwidth=0,
            steps=[
                dict(range=[0, 8], color="rgba(239,83,80,0.12)"),
                dict(range=[8, 12], color="rgba(255,167,38,0.12)"),
                dict(range=[12, 16], color="rgba(100,181,246,0.12)"),
                dict(range=[16, 20], color="rgba(102,187,106,0.12)"),
            ],
            threshold=dict(line=dict(color="#b388ff", width=3), thickness=0.8, value=score),
        ),
        title=dict(text=f"<b>{label}</b>", font=dict(size=16, color="#b388ff")),
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig

# =====================================================================
#  PAGE: Dashboard
# =====================================================================
if page == "📊  Dashboard":
    st.markdown('<div class="hero-title">📊 Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Real-time insights from student data analytics</div>', unsafe_allow_html=True)

    # -- KPI Row --
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{len(df):,}")
    c2.metric("Avg Final Grade (G3)", f"{df['G3'].mean():.1f} / 20")
    c3.metric("Avg Absences", f"{df['absences'].mean():.1f}")
    c4.metric("Model R² Score", f"{r2:.4f}")

    st.markdown("---")

    # -- Charts Row 1 --
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">📈 Final Grade Distribution</div>', unsafe_allow_html=True)
        fig_dist = px.histogram(
            df, x="G3", nbins=21, color_discrete_sequence=["#7c4dff"],
            labels={"G3": "Final Grade (G3)", "count": "Number of Students"},
        )
        fig_dist.update_layout(**PLOTLY_LAYOUT, bargap=0.08,
                               title=None, height=370)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">📚 Study Time vs Final Grade</div>', unsafe_allow_html=True)
        fig_study = px.box(
            df, x="studytime", y="G3", color="studytime",
            color_discrete_sequence=["#4a90d9", "#7c4dff", "#64b5f6", "#b388ff"],
            labels={"studytime": "Weekly Study Time", "G3": "Final Grade (G3)"},
        )
        fig_study.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                                title=None, height=370)
        st.plotly_chart(fig_study, use_container_width=True)

    # -- Charts Row 2 --
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">🔗 G1 vs G3 Comparison</div>', unsafe_allow_html=True)
        fig_g1g3 = px.scatter(
            df, x="G1", y="G3", opacity=0.65,
            color_discrete_sequence=["#64b5f6"],
            labels={"G1": "First Period Grade (G1)", "G3": "Final Grade (G3)"},
        )
        m_g1, b_g1 = np.polyfit(df["G1"], df["G3"], 1)
        x_line_g1 = np.linspace(df["G1"].min(), df["G1"].max(), 50)
        fig_g1g3.add_trace(go.Scatter(
            x=x_line_g1, y=m_g1 * x_line_g1 + b_g1,
            mode="lines", name="Trendline",
            line=dict(color="#b388ff", width=2, dash="dash"),
        ))
        fig_g1g3.update_layout(**PLOTLY_LAYOUT, title=None, height=370, showlegend=False)
        st.plotly_chart(fig_g1g3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">🕐 Recent Predictions</div>', unsafe_allow_html=True)
        if st.session_state.predictions:
            pred_df = pd.DataFrame(st.session_state.predictions[-8:])
            pred_df_display = pred_df[["Timestamp", "Study Time", "G1", "G2",
                                       "Predicted G3", "Grade"]].copy()
            st.dataframe(pred_df_display, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions made yet.  Head over to **Predict Performance** to get started! 🎯")

# =====================================================================
#  PAGE: Predict Performance
# =====================================================================
elif page == "🎯  Predict Performance":
    st.markdown('<div class="hero-title">🎯 Predict Student Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Enter student details to predict their final examination grade</div>', unsafe_allow_html=True)

    # -- Input Form --
    with st.form("prediction_form"):
        st.markdown('<div class="section-header">📝 Student Information</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            studytime = st.slider(
                "📚 Weekly Study Time",
                min_value=1, max_value=4, value=2,
                help="1 = <2 hours · 2 = 2-5 hours · 3 = 5-10 hours · 4 = >10 hours",
            )
            failures = st.slider(
                "❌ Previous Class Failures",
                min_value=0, max_value=4, value=0,
                help="Number of past class failures (0 to 4).",
            )
            absences = st.number_input(
                "📅 Number of Absences",
                min_value=0, max_value=93, value=4, step=1,
                help="Total school absences during the academic year.",
            )

        with col2:
            g1 = st.slider(
                "📝 First Period Marks (G1)",
                min_value=0, max_value=20, value=12,
                help="Grade obtained in the first evaluation period (0–20).",
            )
            g2 = st.slider(
                "📝 Second Period Marks (G2)",
                min_value=0, max_value=20, value=13,
                help="Grade obtained in the second evaluation period (0–20).",
            )

        st.markdown("")
        col_btn, col_reset = st.columns([3, 1])
        with col_btn:
            predict_btn = st.form_submit_button(
                "🚀  PREDICT FINAL PERFORMANCE", use_container_width=True
            )
        with col_reset:
            reset_btn = st.form_submit_button("🔄 Reset", use_container_width=True)

    # -- Handle reset (re-run with defaults is enough; Streamlit re-renders) --
    if reset_btn:
        st.rerun()

    # -- Handle prediction --
    if predict_btn:
        with st.spinner("Running prediction model…"):
            predicted = predict_grade(model, studytime, failures, absences, g1, g2)

        label, css_class = grade_label(predicted)

        # Store prediction
        st.session_state.predictions.append({
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Study Time": studytime,
            "Failures": failures,
            "Absences": absences,
            "G1": g1,
            "G2": g2,
            "Predicted G3": predicted,
            "Grade": label,
        })

        # -- Result display --
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.plotly_chart(build_gauge(predicted), use_container_width=True)

        with res_col2:
            st.markdown(
                f"""
                <div class="prediction-result">
                    <div class="prediction-score">{predicted}</div>
                    <div class="prediction-label">Predicted Final Grade out of 20</div>
                    <div class="grade-badge {css_class}">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                f"""
                <div class="card card-blue" style="text-align:left; font-size:.9rem;">
                    <b>Input Summary</b><br>
                    📚 Study Time: <b>{studytime}</b> &nbsp;|&nbsp;
                    ❌ Failures: <b>{failures}</b> &nbsp;|&nbsp;
                    📅 Absences: <b>{absences}</b><br>
                    📝 G1: <b>{g1}</b> &nbsp;|&nbsp;
                    📝 G2: <b>{g2}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="disclaimer">⚠️ <b>Disclaimer:</b> This prediction is '
            "a statistical estimate generated by a Linear Regression model trained "
            "on historical data. It is <b>not</b> a guaranteed result and should "
            "not replace professional academic evaluation.</div>",
            unsafe_allow_html=True,
        )

    # -- Prediction History --
    if st.session_state.predictions:
        st.markdown("---")
        st.markdown('<div class="section-header">🕐 Prediction History</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        dl_col, clr_col, _ = st.columns([1, 1, 2])
        with dl_col:
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download Report (CSV)",
                data=csv_data,
                file_name="prediction_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with clr_col:
            if st.button("🗑️  Clear History", use_container_width=True):
                st.session_state.predictions = []
                st.rerun()

# =====================================================================
#  PAGE: Student Analytics
# =====================================================================
elif page == "📈  Student Analytics":
    st.markdown('<div class="hero-title">📈 Student Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">In-depth analysis of student performance factors</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Grade Distribution", "📚 Study Time", "📅 Absences",
        "❌ Failures", "🔗 Correlations",
    ])

    # ── Tab 1 ───────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Grade Distribution (G1, G2, G3)</div>', unsafe_allow_html=True)
        fig_grades = go.Figure()
        for col, name, color in [("G1", "Period 1", "#4a90d9"),
                                  ("G2", "Period 2", "#7c4dff"),
                                  ("G3", "Final", "#64b5f6")]:
            fig_grades.add_trace(go.Histogram(
                x=df[col], name=name, marker_color=color, opacity=0.7, nbinsx=21,
            ))
        fig_grades.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
                                 xaxis_title="Grade", yaxis_title="Count",
                                 height=420, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_grades, use_container_width=True)

        # Pass / Fail breakdown (pass = G3 >= 10)
        pass_count = int((df["G3"] >= 10).sum())
        fail_count = len(df) - pass_count
        fig_pf = px.pie(
            names=["Pass (≥ 10)", "Fail (< 10)"],
            values=[pass_count, fail_count],
            color_discrete_sequence=["#66bb6a", "#ef5350"],
            hole=0.5,
        )
        fig_pf.update_layout(**PLOTLY_LAYOUT, height=350,
                             title=dict(text="Pass / Fail Ratio", font=dict(size=16)))
        st.plotly_chart(fig_pf, use_container_width=True)

    # ── Tab 2 ───────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Study Time Analysis</div>', unsafe_allow_html=True)
        study_avg = df.groupby("studytime")["G3"].mean().reset_index()
        study_avg.columns = ["Study Time", "Average G3"]
        fig_st = px.bar(
            study_avg, x="Study Time", y="Average G3",
            color="Average G3",
            color_continuous_scale=["#4a90d9", "#7c4dff"],
            text_auto=".1f",
        )
        fig_st.update_layout(**PLOTLY_LAYOUT, height=400,
                             coloraxis_showscale=False)
        fig_st.update_traces(textposition="outside")
        st.plotly_chart(fig_st, use_container_width=True)

        fig_violin = px.violin(
            df, x="studytime", y="G3", color="studytime", box=True, points="all",
            color_discrete_sequence=["#4a90d9", "#7c4dff", "#64b5f6", "#b388ff"],
            labels={"studytime": "Study Time", "G3": "Final Grade"},
        )
        fig_violin.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=420)
        st.plotly_chart(fig_violin, use_container_width=True)

    # ── Tab 3 ───────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Absences vs Final Grade</div>', unsafe_allow_html=True)
        fig_abs = px.scatter(
            df, x="absences", y="G3", opacity=0.6,
            color_discrete_sequence=["#64b5f6"],
            labels={"absences": "Number of Absences", "G3": "Final Grade (G3)"},
        )
        m_abs, b_abs = np.polyfit(df["absences"], df["G3"], 1)
        x_line_abs = np.linspace(df["absences"].min(), df["absences"].max(), 50)
        fig_abs.add_trace(go.Scatter(
            x=x_line_abs, y=m_abs * x_line_abs + b_abs,
            mode="lines", name="Trendline",
            line=dict(color="#b388ff", width=2, dash="dash"),
        ))
        fig_abs.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
        st.plotly_chart(fig_abs, use_container_width=True)

        # Binned analysis
        df_tmp = df.copy()
        df_tmp["Absence Group"] = pd.cut(
            df_tmp["absences"],
            bins=[0, 5, 10, 20, 100],
            labels=["0-5", "6-10", "11-20", "21+"],
            right=True,
        )
        abs_avg = df_tmp.groupby("Absence Group", observed=True)["G3"].mean().reset_index()
        fig_abs_bar = px.bar(
            abs_avg, x="Absence Group", y="G3", text_auto=".1f",
            color_discrete_sequence=["#7c4dff"],
            labels={"G3": "Average Final Grade"},
        )
        fig_abs_bar.update_layout(**PLOTLY_LAYOUT, height=370)
        fig_abs_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_abs_bar, use_container_width=True)

    # ── Tab 4 ───────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Previous Failures vs Final Grade</div>', unsafe_allow_html=True)
        fail_avg = df.groupby("failures")["G3"].mean().reset_index()
        fail_avg.columns = ["Failures", "Average G3"]
        fig_fail = px.bar(
            fail_avg, x="Failures", y="Average G3",
            color="Average G3",
            color_continuous_scale=["#ef5350", "#66bb6a"],
            text_auto=".1f",
        )
        fig_fail.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
        fig_fail.update_traces(textposition="outside")
        st.plotly_chart(fig_fail, use_container_width=True)

        fig_fail_box = px.box(
            df, x="failures", y="G3",
            color_discrete_sequence=["#b388ff"],
            labels={"failures": "Number of Failures", "G3": "Final Grade"},
        )
        fig_fail_box.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_fail_box, use_container_width=True)

    # ── Tab 5 ───────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = ["studytime", "failures", "absences", "G1", "G2", "G3"]
        corr = df[corr_cols].corr()

        fig_heat = px.imshow(
            corr,
            x=corr_cols, y=corr_cols,
            color_continuous_scale=["#0f1b2d", "#4a90d9", "#7c4dff", "#b388ff"],
            text_auto=".2f",
            aspect="auto",
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=480,
                               title=dict(text="Pearson Correlation Matrix",
                                          font=dict(size=16)))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown(
            """
            <div class="card card-purple">
                <b>Key Insights:</b><br>
                • G2 (second period grade) has the strongest correlation with G3 (final grade).<br>
                • G1 also strongly correlates with G3 — early performance is predictive.<br>
                • More past failures tend to correlate with lower final grades.<br>
                • Study time shows a mild positive correlation with final grades.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =====================================================================
#  PAGE: Dataset Explorer
# =====================================================================
elif page == "🗃️  Dataset Explorer":
    st.markdown('<div class="hero-title">🗃️ Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Browse, filter and understand the student dataset</div>', unsafe_allow_html=True)

    # -- Quick stats --
    qs1, qs2, qs3, qs4 = st.columns(4)
    qs1.metric("Rows", f"{df.shape[0]:,}")
    qs2.metric("Columns", f"{df.shape[1]}")
    qs3.metric("Missing Values", f"{int(df.isnull().sum().sum())}")
    qs4.metric("Duplicate Rows", f"{int(df.duplicated().sum())}")

    st.markdown("---")

    # -- Tabs --
    dt1, dt2, dt3, dt4 = st.tabs([
        "📋 Preview", "📊 Statistics", "🔍 Filter & Search", "⬇️ Download",
    ])

    with dt1:
        st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=420)

        with st.expander("📝 Column Descriptions"):
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Unique": df.nunique().values,
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)

    with dt2:
        st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().T.round(2), use_container_width=True)

        st.markdown('<div class="section-header">Missing Values per Column</div>', unsafe_allow_html=True)
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column", "Missing"]
        mv["% Missing"] = (mv["Missing"] / len(df) * 100).round(2)
        st.dataframe(mv[mv["Missing"] > 0] if mv["Missing"].sum() > 0
                     else mv.head(10), use_container_width=True, hide_index=True)
        if mv["Missing"].sum() == 0:
            st.success("✅ No missing values found in the dataset!")

    with dt3:
        st.markdown('<div class="section-header">Filter Student Records</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            g3_range = st.slider("Final Grade (G3) Range", 0, 20, (0, 20))
        with fc2:
            st_filter = st.multiselect("Study Time", options=sorted(df["studytime"].unique()),
                                       default=sorted(df["studytime"].unique()))
        with fc3:
            fail_filter = st.multiselect("Failures", options=sorted(df["failures"].unique()),
                                         default=sorted(df["failures"].unique()))

        filtered = df[
            (df["G3"] >= g3_range[0]) & (df["G3"] <= g3_range[1]) &
            (df["studytime"].isin(st_filter)) &
            (df["failures"].isin(fail_filter))
        ]
        st.info(f"Showing **{len(filtered):,}** of **{len(df):,}** records")
        st.dataframe(filtered, use_container_width=True, height=380)

    with dt4:
        st.markdown('<div class="section-header">Download Dataset</div>', unsafe_allow_html=True)
        col_dl1, col_dl2, _ = st.columns([1, 1, 2])
        with col_dl1:
            st.download_button(
                "⬇️ Download Full Dataset (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="student_performance_full.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            if len(filtered) < len(df):
                st.download_button(
                    "⬇️ Download Filtered Data (CSV)",
                    data=filtered.to_csv(index=False).encode("utf-8"),
                    file_name="student_performance_filtered.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# =====================================================================
#  PAGE: Model Information
# =====================================================================
elif page == "🤖  Model Information":
    st.markdown('<div class="hero-title">🤖 Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Understand the Machine Learning behind the predictions</div>', unsafe_allow_html=True)

    # -- Quick model metrics --
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm", "Linear Regression")
    m2.metric("R² Score", f"{r2:.4f}")
    m3.metric("MAE", f"{mae:.4f}")
    m4.metric("Train / Test Split", "80 : 20")

    st.markdown("---")

    # -- ML concepts --
    st.markdown('<div class="section-header">📖 Machine Learning Concepts</div>', unsafe_allow_html=True)

    with st.expander("🧠 What is Machine Learning?", expanded=True):
        st.markdown(
            """
            **Machine Learning (ML)** is a branch of Artificial Intelligence that enables
            computers to learn from data and make predictions or decisions without being
            explicitly programmed for every scenario.

            In this project, the model *learns* the relationship between student attributes
            (study time, failures, absences, G1, G2) and their final grade (G3) from
            historical data, then uses that learned relationship to predict outcomes for
            new students.
            """
        )

    with st.expander("📐 What is Linear Regression?"):
        st.markdown(
            """
            **Linear Regression** fits a straight-line (or hyper-plane in multiple
            dimensions) relationship between one or more input features and a continuous
            target variable.

            The model equation:

            $$\\hat{y} = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\cdots + \\beta_n x_n$$

            where $\\hat{y}$ is the predicted grade, $x_i$ are the input features, and
            $\\beta_i$ are the learned coefficients.

            It minimizes the **sum of squared residuals** (differences between predicted
            and actual values) to find the best-fitting line.
            """
        )

    with st.expander("🔀 How Training and Testing Works"):
        st.markdown(
            """
            The dataset is split into two non-overlapping subsets:

            | Subset | Purpose | Size |
            |--------|---------|------|
            | **Training set** | Used to fit the model (learn coefficients) | 80 % |
            | **Testing set**  | Used to evaluate how well the model generalizes | 20 % |

            We use `random_state=42` so the split is reproducible — anyone running
            this code gets the same split.

            **Why split?**  If we evaluated the model on the same data it was trained on,
            we would get an overly optimistic estimate of performance.  The test set
            simulates *unseen* students.
            """
        )

    with st.expander("📊 Evaluation Metrics"):
        st.markdown(
            f"""
            | Metric | Value | Meaning |
            |--------|-------|---------|
            | **MAE** (Mean Absolute Error) | **{mae:.4f}** | On average, predictions deviate from the true grade by ~{mae:.2f} marks. |
            | **R² Score** | **{r2:.4f}** | The model explains ~{r2*100:.1f} % of the variance in final grades. |

            * **MAE** = average of |actual − predicted| across all test samples.
            * **R²** = 1 means perfect prediction; 0 means the model is no better than
              always predicting the mean.
            """
        )

    st.markdown("---")

    # -- Actual vs Predicted scatter --
    st.markdown('<div class="section-header">📉 Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=y_test.values, y=y_pred,
        mode="markers", marker=dict(color="#7c4dff", size=8, opacity=0.7),
        name="Predictions",
    ))
    fig_ap.add_trace(go.Scatter(
        x=[0, 20], y=[0, 20],
        mode="lines", line=dict(color="#ef5350", dash="dash", width=2),
        name="Perfect Prediction",
    ))
    fig_ap.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Actual G3",
        yaxis_title="Predicted G3",
        height=440,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_ap, use_container_width=True)

    # -- Residuals --
    st.markdown('<div class="section-header">📊 Residual Distribution</div>', unsafe_allow_html=True)
    residuals = y_test.values - y_pred
    fig_res = px.histogram(
        x=residuals, nbins=25,
        color_discrete_sequence=["#4a90d9"],
        labels={"x": "Residual (Actual − Predicted)", "count": "Count"},
    )
    fig_res.update_layout(**PLOTLY_LAYOUT, height=370)
    st.plotly_chart(fig_res, use_container_width=True)

    # -- Feature coefficients --
    st.markdown('<div class="section-header">⚖️ Feature Coefficients</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature": result["feature_cols"],
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", key=abs, ascending=True)

    fig_coef = px.bar(
        coef_df, x="Coefficient", y="Feature", orientation="h",
        color="Coefficient",
        color_continuous_scale=["#ef5350", "#90a4ae", "#66bb6a"],
        color_continuous_midpoint=0,
        text_auto=".3f",
    )
    fig_coef.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_showscale=False)
    fig_coef.update_traces(textposition="outside")
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown(
        """
        <div class="card card-purple">
            <b>Interpreting Coefficients:</b><br>
            • A <b>positive</b> coefficient means the feature increases the predicted grade.<br>
            • A <b>negative</b> coefficient means the feature decreases the predicted grade.<br>
            • <b>G2</b> (second period grade) has the largest positive weight — it is the
            strongest predictor of the final grade.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =====================================================================
#  PAGE: About Project
# =====================================================================
elif page == "ℹ️  About Project":
    st.markdown('<div class="hero-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Student Performance Prediction using Machine Learning</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card card-blue">
            <h3 style="margin-top:0;">🎓 Project Title</h3>
            <p style="font-size:1.1rem; font-weight:600;">
                Student Performance Prediction System using Linear Regression
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card card-purple">
            <h3 style="margin-top:0;">📄 Abstract</h3>
            <p>
                This project develops an interactive web-based application that predicts
                students' final examination grades in Mathematics using Machine Learning.
                The system is trained on the publicly available UCI Student Performance
                dataset and employs Linear Regression to model the relationship between
                key academic and behavioural factors — study time, previous failures,
                absences, and intermediate period grades — and the final examination
                outcome.  A modern, visually polished dashboard built with Streamlit
                allows users to explore the data, run predictions, and understand the
                underlying model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="card">
                <h3 style="margin-top:0;">🔍 Problem Statement</h3>
                <p>
                    Educational institutions need early-warning mechanisms to identify
                    students at risk of underperforming.  Manual assessment is
                    time-consuming and subjective.  This project demonstrates how
                    historical data and simple Machine Learning can provide timely,
                    data-driven performance estimates that educators may use as one of
                    several inputs in their evaluation process.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
                <h3 style="margin-top:0;">🎯 Objectives</h3>
                <ul>
                    <li>Build a predictive model for student final grades.</li>
                    <li>Create an interactive and visually appealing web dashboard.</li>
                    <li>Enable educators to explore data-driven insights.</li>
                    <li>Demonstrate the end-to-end ML workflow in a beginner-friendly manner.</li>
                    <li>Evaluate model performance with standard regression metrics.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="card">
                <h3 style="margin-top:0;">🛠️ Technologies Used</h3>
                <p>
                    <span class="tech-badge">Python</span>
                    <span class="tech-badge">Streamlit</span>
                    <span class="tech-badge">Pandas</span>
                    <span class="tech-badge">NumPy</span>
                    <span class="tech-badge">Scikit-learn</span>
                    <span class="tech-badge">Plotly</span>
                    <span class="tech-badge">Matplotlib</span>
                    <span class="tech-badge">HTML / CSS</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
                <h3 style="margin-top:0;">⚙️ Algorithm</h3>
                <p>
                    <b>Linear Regression</b> — a supervised learning algorithm that
                    models the linear relationship between independent features and a
                    continuous target variable by minimising the sum of squared
                    residuals.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
                <h3 style="margin-top:0;">📦 Dataset Source</h3>
                <p>
                    UCI Machine Learning Repository — <b>Student Performance Dataset</b><br>
                    <a href="https://archive.ics.uci.edu/dataset/320/student+performance"
                       target="_blank">
                        https://archive.ics.uci.edu/dataset/320/student+performance
                    </a><br>
                    File used: <code>student-mat.csv</code> (Mathematics course).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card card-success">
            <h3 style="margin-top:0;">✅ Expected Outcomes</h3>
            <ul>
                <li>A fully functional web application with interactive analytics.</li>
                <li>Accurate grade prediction using validated ML metrics.</li>
                <li>Visual insights into factors influencing student performance.</li>
                <li>A reusable project template for similar educational data science tasks.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
