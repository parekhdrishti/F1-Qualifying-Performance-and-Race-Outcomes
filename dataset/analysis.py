import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# Load cleaned data
df = pd.read_csv("data/cleaned_f1.csv")

print("Dataset loaded:", df.shape)

# Correlation analysis: Qualifying position vs Finish position
# We will use Spearman's rank correlation since positions are ordinal   
# Spearman correlation

spearman_corr, _ = stats.spearmanr(df["quali_pos"], df["finish_position"])

# Pearson correlation
pearson_corr, _ = stats.pearsonr(df["quali_pos"], df["finish_position"])

print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")

#MAE: Mean Absolute Error Prediction error between qualifying position and finish position  
#(On average, how many positions drivers gain/lose from qualifying to finish)
mae = mean_absolute_error(df["finish_position"], df["quali_pos"])

print(f"Mean Absolute Error (MAE): {mae:.4f}")

from sklearn.linear_model import LinearRegression
import numpy as np

# ── LINEAR REGRESSION ──────────────────────────────────────────────
X = df[["quali_pos"]]
y = df["finish_position"]

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)

print(f"Linear Regression Coefficient (slope): {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R² Score: {r_squared:.4f}")

# What this means:
# slope ≈ 0.66 means for every 1 position worse in qualifying,
# a driver finishes ~0.66 positions worse on average

from scipy.stats import chi2_contingency

# -------------------------------
# Summary statistics
# -------------------------------
summary_stats = df[["quali_pos", "finish_position", "pos_change"]].describe().T
summary_stats = summary_stats[["mean", "50%", "std", "min", "25%", "75%", "max"]]
summary_stats = summary_stats.rename(columns={"50%": "median"})

print("\nSummary Statistics:")
print(summary_stats)

# -------------------------------
# Era-wise correlation analysis
# -------------------------------
era_results = []

for era_name, era_df in df.groupby("era"):
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

print("\nEra-wise Results:")
print(era_results_df)

# -------------------------------
# Chi-square test: grid group vs podium finish
# -------------------------------
contingency_table = pd.crosstab(df["grid_group"], df["is_podium"])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("\nChi-square Test: Grid Group vs Podium Finish")
print("Contingency Table:")
print(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")

#Scatter plot: Qualifying position vs Finish position
plt.figure()
plt.scatter(df["quali_pos"], df["finish_position"], alpha=0.3)

plt.xlabel("Qualifying Position")
plt.ylabel("Finish Position")
plt.title("Qualifying vs Race Finish")

plt.show()
#The scatter plot shows a clear upward trend, meaning better qualifying positions generally lead to better race finishes, although there is some variability

#Podium Probability
podium_prob = df.groupby("grid_group")["is_podium"].mean()

print("\nPodium Probability by Grid Group:")
print(podium_prob)

#Counts for intrepretation
counts = df.groupby("grid_group")["is_podium"].count()

print("\nNumber of Drivers in Each Group:")
print(counts)

#Histogram  -Position changes
#Interpretation
#The distribution of position changes is centered around zero, indicating that most drivers finish close to their starting positions. Large gains or losses are less frequent, reinforcing the idea that starting position is an important factor in race outcomes.

plt.figure(figsize=(7, 5))

sns.histplot(df["pos_change"], bins=30)

plt.title("Distribution of Positions Gained/Lost")
plt.xlabel("Position Change (Grid - Finish)")
plt.ylabel("Frequency")

plt.show()
#Most drivers finish close to their starting position, as shown by the concentration around zero. Large gains or losses are less common.

#Podium probability by grid group
podium_stats = (
    df.groupby("grid_group")["is_podium"]
    .agg(podium_prob="mean", count="count")
    .reset_index()
)

print("\nPodium Probability by Grid Group:")
print(podium_stats)
plt.figure(figsize=(7, 5))

sns.barplot(
    data=podium_stats,
    x="grid_group",
    y="podium_prob"
)

plt.title("Podium Probability by Grid Group")
plt.xlabel("Grid Group")
plt.ylabel("Probability of Podium Finish")

plt.show()
#The probability of finishing on the podium decreases sharply as starting position worsens, highlighting the importance of qualifying performance.
#This analysis shows that qualifying performance is a strong predictor of race outcomes in Formula 1. Drivers who start in higher grid positions are significantly more likely to finish in top positions, especially on the podium. While some variability exists due to race conditions and strategy, the overall trend confirms that starting position plays a crucial role in determining race success.

