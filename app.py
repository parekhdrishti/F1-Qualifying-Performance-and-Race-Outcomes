import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.metrics import mean_absolute_error
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(
    page_title="F1 Qualifying vs Race Outcomes",
    layout="wide"
)

st.title("\U0001F3CE\uFE0F F1 Qualifying vs Race Outcomes Dashboard")
st.write("Analysis of qualifying position vs race finish from 2000-2024.")

with st.expander("About this dashboard"):
    st.write(
        "This dashboard analyzes how qualifying position relates to race finish in Formula 1 "
        "from 2000 to 2024. "
        "The dataset was cleaned by merging race and qualifying data, removing DNFs, and "
        "creating derived variables "
        "such as era, grid group, and position change."
    )
# ------------------------------
# Load data
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_f1.csv")

df = load_data()

# ------------------------------
# Sidebar filters
# ------------------------------
st.sidebar.header("Filters")

years = sorted(df["year"].dropna().unique())
year_min = int(min(years))
year_max = int(max(years))

selected_year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

selected_years = list(range(selected_year_range[0], selected_year_range[1] + 1))

teams = sorted(df["team_name"].dropna().unique())
selected_teams = st.sidebar.multiselect(
    "Select Team(s) - leave empty for all",
    teams,
    default=[]
)

drivers = sorted(df["driver_name"].dropna().unique())
selected_drivers = st.sidebar.multiselect(
    "Select Driver(s) - leave empty for all",
    drivers,
    default=[]
)

eras = sorted(df["era"].dropna().unique())
selected_eras = st.sidebar.multiselect(
    "Select Era(s)",
    eras,
    default=eras
)

filtered_df = df[
    (df["year"].isin(selected_years)) &
    (df["era"].isin(selected_eras))
].copy()

if selected_teams:
    filtered_df = filtered_df[filtered_df["team_name"].isin(selected_teams)]

if selected_drivers:
    filtered_df = filtered_df[filtered_df["driver_name"].isin(selected_drivers)]

st.sidebar.caption("Tip: Leave team and driver blank to view the full dataset.")

st.write(f"Filtered dataset size: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")

# Stop if no data
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ------------------------------
# Key metrics
# ------------------------------
spearman_corr, _ = stats.spearmanr(
    filtered_df["quali_pos"],
    filtered_df["finish_position"]
)

pearson_corr, _ = stats.pearsonr(
    filtered_df["quali_pos"],
    filtered_df["finish_position"]
)

mae = mean_absolute_error(
    filtered_df["finish_position"],
    filtered_df["quali_pos"]
)

# Summary statistics table
summary_stats = filtered_df[["quali_pos", "finish_position", "pos_change"]].describe().T
summary_stats = summary_stats[["mean", "50%", "std", "min", "25%", "75%", "max"]]
summary_stats = summary_stats.rename(columns={"50%": "median"})

# Era-wise metrics
era_results = []
for era_name, era_df in filtered_df.groupby("era"):
    if len(era_df) > 1:
        spearman_corr_era, _ = stats.spearmanr(era_df["quali_pos"], era_df["finish_position"])
        pearson_corr_era, _ = stats.pearsonr(era_df["quali_pos"], era_df["finish_position"])
        mae_era = mean_absolute_error(era_df["finish_position"], era_df["quali_pos"])

        era_results.append({
            "era": era_name,
            "count": len(era_df),
            "spearman_corr": spearman_corr_era,
            "pearson_corr": pearson_corr_era,
            "mae": mae_era
        })

era_results_df = pd.DataFrame(era_results)

contingency_table = pd.crosstab(filtered_df["grid_group"], filtered_df["is_podium"])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

col1, col2, col3 = st.columns(3)

col1.metric("Spearman Correlation", f"{spearman_corr:.4f}")
col1.caption("Measures the rank-based relationship between qualifying and finish position.")

col2.metric("Pearson Correlation", f"{pearson_corr:.4f}")
col2.caption("Measures the linear relationship between qualifying and finish position.")

col3.metric("MAE", f"{mae:.2f} positions")
col3.caption("Average gap between qualifying position and final race finish.")


# Chi-square test
contingency_table = pd.crosstab(filtered_df["grid_group"], filtered_df["is_podium"])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Qualifying vs Finish",
    "Position Change",
    "Podium Probability",
    "Summary & Era Analysis"
])
# ------------------------------
# Tab 1: Scatter plot
# ------------------------------
with tab1:
    st.subheader("Qualifying Position vs Finish Position")

    color_mode = st.radio(
        "Scatter plot coloring",
        ["Single color", "Color by team"],
        horizontal=True
    )

    if color_mode == "Color by team":
        fig = px.scatter(
            filtered_df,
            x="quali_pos",
            y="finish_position",
            color="team_name",
            hover_data=["year", "driver_name", "team_name", "grid_position", "finish_position"],
            title="Qualifying Position vs Race Finish",
            labels={
                "quali_pos": "Qualifying Position",
                "finish_position": "Finish Position",
                "team_name": "Team"
            },
            opacity=0.6,
            trendline="ols"
        )
    else:
        fig = px.scatter(
            filtered_df,
            x="quali_pos",
            y="finish_position",
            hover_data=["year", "driver_name", "team_name", "grid_position", "finish_position"],
            title="Qualifying Position vs Race Finish",
            labels={
                "quali_pos": "Qualifying Position",
                "finish_position": "Finish Position"
            },
            opacity=0.45,
            trendline="ols"
        )

        fig.update_traces(marker=dict(color="#4C78A8", size=7), selector=dict(mode="markers"))

    fig.update_layout(
        legend_title_text="Team",
        xaxis=dict(dtick=1),
        yaxis=dict(dtick=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "This chart shows the relationship between qualifying position and finishing position. "
        "A strong upward pattern means drivers who qualify better usually finish better."
        "The trend line helps highlight the overall positive relationship, while team-color "
        "mode is useful for comparing constructors."
    )

# ------------------------------
# Tab 2: Histogram
# ------------------------------
with tab2:
    st.subheader("Distribution of Positions Gained or Lost")

    fig = px.histogram(
        filtered_df,
        x="pos_change",
        nbins=30,
        title="Distribution of Position Change",
        labels={
            "pos_change": "Position Change (Grid - Finish)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "Most drivers finish close to where they started. Large gains or losses happen less "
        "often."
    )

    st.subheader("Position Change by Grid Group")

    fig_box = px.box(
        filtered_df,
        x="grid_group",
        y="pos_change",
        color="grid_group",
        title="Position Change by Grid Group",
        labels={
            "grid_group": "Grid Group",
            "pos_change": "Position Change (Grid - Finish)"
        }
    )

    st.plotly_chart(fig_box, use_container_width=True)

    st.write(
        "This boxplot compares how many positions drivers gain or lose depending on where "
        "they started. "
        "It helps show spread, median, and unusual race outcomes."
    )
# ------------------------------
# Tab 3: Podium probability
# ------------------------------
with tab3:
    st.subheader("Podium Probability by Grid Group")

    podium_stats = (
        filtered_df.groupby("grid_group")["is_podium"]
        .agg(podium_probability="mean", count="count")
        .reset_index()
    )

    fig = px.bar(
        podium_stats,
        x="grid_group",
        y="podium_probability",
        text="podium_probability",
        title="Podium Probability by Grid Group",
        labels={
            "grid_group": "Grid Group",
            "podium_probability": "Probability of Podium Finish"
        }
    )

    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(podium_stats)

    st.write(
        "Drivers starting in the Top 3 have the highest chance of finishing on the podium."
    )

    # ------------------------------
# Tab 4: Summary and Era Analysis
# ------------------------------
with tab4:
    st.subheader("Summary Statistics")
    st.dataframe(summary_stats)

    st.write(
        "These descriptive statistics summarize the distributions of qualifying position, "
        "finish position, "
        "and positions gained or lost."
    )

    st.subheader("Era-wise Comparison")
    st.dataframe(era_results_df)

    if not era_results_df.empty:
        fig_era = px.bar(
            era_results_df,
            x="era",
            y="spearman_corr",
            text="spearman_corr",
            title="Spearman Correlation by Era",
            labels={
                "era": "Era",
                "spearman_corr": "Spearman Correlation"
            }
        )

        fig_era.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig_era, use_container_width=True)

    st.write(
        "This section compares how strongly qualifying predicts race finish across different "
        "Formula 1 eras."
    )

    st.subheader("Chi-square Test: Grid Group vs Podium Finish")
    st.write("Contingency Table:")
    st.dataframe(contingency_table)

    st.write(f"Chi-square statistic: {chi2:.4f}")
    st.write(f"p-value: {p_value:.6f}")
    st.write(f"Degrees of freedom: {dof}")

    st.write(
        "A very small p-value would indicate that podium outcomes depend strongly on starting "
        "grid group."
    )

    # Limitations note
    st.info(
        "Note: This analysis excludes DNFs and incomplete records, so the results describe "
        "finishing outcomes "
        "among classified finishers rather than all race entrants."
    )
from sklearn.linear_model import LinearRegression

# --- LINEAR REGRESSION -----------------------------------------------
st.subheader("Linear Regression: Qualifying vs Finish Position")

X = filtered_df[["quali_pos"]]
y = filtered_df["finish_position"]

lr_model = LinearRegression()
lr_model.fit(X, y)

slope = lr_model.coef_[0]
intercept = lr_model.intercept_
r_squared = lr_model.score(X, y)

col1, col2, col3 = st.columns(3)
col1.metric("Slope", f"{slope:.4f}")
col1.caption("Position change per qualifying spot dropped.")

col2.metric("Intercept", f"{intercept:.4f}")
col2.caption("Predicted finish if qualifying position were 0.")

col3.metric("R\u00b2 Score", f"{r_squared:.4f}")
col3.caption("Proportion of finish position variance explained by qualifying.")

st.write(
    f"For every 1 position worse in qualifying, a driver is predicted to "
    f"finish {slope:.2f} positions worse in the race. "
    f"The R\u00b2 of {r_squared:.4f} means qualifying position explains "
    f"{r_squared*100:.1f}% of the variance in finishing position."
)
