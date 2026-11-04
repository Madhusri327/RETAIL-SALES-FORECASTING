import pandas as pd
import streamlit as st

from dataset.train import train_model


st.set_page_config(
    page_title="Retail Sales Forecast",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        --ink: #202a25;
        --muted: #6b7770;
        --paper: #f5f6f1;
        --surface: #ffffff;
        --line: #dfe5dc;
        --green: #245c45;
        --mint: #dcebe1;
        --coral: #c7654b;
    }
    .stApp { background: linear-gradient(180deg, #edf2eb 0, var(--paper) 280px); color: var(--ink); }
    [data-testid="stSidebar"] { background: #eef1e9; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', 'Trebuchet MS', sans-serif !important; color: var(--ink); letter-spacing: 0 !important; }
    p, label, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
    h1 { font-size: 2.25rem !important; font-weight: 600 !important; margin-bottom: .15rem !important; }
    .eyebrow { color: var(--green); font: 700 .72rem 'DM Sans', sans-serif; text-transform: uppercase; letter-spacing: .12em; }
    .subtle { color: var(--muted); font-size: .94rem; }
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.15rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    div.stButton > button, [data-testid="stFormSubmitButton"] > button { background: var(--green); color: white; border: 0; border-radius: 5px; font-weight: 600; min-height: 2.8rem; }
    div.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: #174630; color: white; border: 0; }
    [data-testid="stAlert"] { border-radius: 5px; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return train_model()


def season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


st.markdown('<div class="eyebrow">Retail intelligence / weekly demand</div>', unsafe_allow_html=True)
st.title("Sales forecast")

with st.spinner("Training the forecasting model on store history…"):
    try:
        model, df, metrics = get_model_bundle()
    except Exception as error:
        st.error(f"Could not prepare the forecasting model: {error}")
        st.stop()

last_date = df["date"].max()
next_week = (last_date + pd.Timedelta(weeks=1)).date()
stores = sorted(df["store_id"].unique().tolist())
holiday_values = sorted(
    {
        str(value)
        for value in df["holiday_name"].dropna().unique()
        if str(value).strip().lower() not in {"none", "nan", ""}
    }
)

with st.sidebar:
    st.markdown("### Forecast inputs")
    with st.form("forecast_inputs"):
        store_id = st.selectbox("Store", stores, format_func=lambda value: f"Store {value}")
        departments = sorted(
            df.loc[df["store_id"] == store_id, "department"].unique().tolist()
        )
        department = st.selectbox("Department", departments)
        forecast_date = st.date_input(
            "Forecast week",
            value=next_week,
            min_value=next_week,
        )
        holiday_choice = st.selectbox("Holiday", ["No holiday", *holiday_values])
        submitted = st.form_submit_button("Run forecast", use_container_width=True)
    st.caption(f"History available through {last_date:%B %d, %Y}")

summary_cols = st.columns(3)
summary_cols[0].metric("Stores", f"{df['store_id'].nunique():,}")
summary_cols[1].metric("Departments", f"{df['department'].nunique():,}")
summary_cols[2].metric(
    "Holdout MAE",
    f"${metrics['mae'] / 1000:.1f}k",
    help=f"Exact mean absolute error: ${metrics['mae']:,.2f}",
)

history = df.loc[
    (df["store_id"] == store_id) & (df["department"] == department)
].sort_values("date").tail(26)

if submitted:
    forecast_row = (
        df.loc[df["store_id"] == store_id]
        .sort_values("date")
        .tail(1)
        .copy()
    )
    forecast_row["date"] = pd.Timestamp(forecast_date)
    forecast_row["department"] = department
    forecast_row["year"] = forecast_row["date"].dt.year
    forecast_row["week_of_year"] = forecast_row["date"].dt.isocalendar().week.astype(int)
    forecast_row["season"] = season_for_month(forecast_date.month)
    forecast_row["holiday_name"] = "None" if holiday_choice == "No holiday" else holiday_choice
    forecast_row["is_holiday"] = int(holiday_choice != "No holiday")

    model_input = forecast_row.drop(columns=["weekly_sales", "date"])
    forecast_value = float(model.predict(model_input)[0])
    recent_average = float(history.tail(8)["weekly_sales"].mean())
    change = forecast_value / recent_average - 1 if recent_average else 0

    st.divider()
    result_cols = st.columns([1, 2])
    with result_cols[0]:
        st.metric(
            f"Projected sales · {forecast_date:%b %d}",
            f"${forecast_value:,.0f}",
            f"{change:+.1%} vs recent 8-week average",
        )
        st.caption(
            f"Model: Random Forest · Validation RMSE ${metrics['rmse']:,.0f}"
        )
    with result_cols[1]:
        st.markdown("#### Recent weekly sales")
        st.line_chart(
            history.set_index("date")["weekly_sales"],
            height=280,
            color="#245c45",
        )
    st.caption(
        f"Forecast conditions are based on the latest observed store data ({last_date:%b %d, %Y}); "
        "future weather, markdowns, and economic indicators are not known and are held at their latest values."
    )
else:
    st.divider()
    st.markdown("#### Recent weekly sales")
    st.line_chart(
        history.set_index("date")["weekly_sales"],
        height=320,
        color="#245c45",
    )
    st.caption(f"Selected store history through {last_date:%b %d, %Y}.")