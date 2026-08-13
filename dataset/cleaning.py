import pandas as pd
from scipy import stats

# Load data
path = "data/"

races = pd.read_csv(path + "races.csv")
results = pd.read_csv(path + "results.csv")
qualifying = pd.read_csv(path + "qualifying.csv")
status = pd.read_csv(path + "status.csv")
drivers = pd.read_csv(path + "drivers.csv")
constructors = pd.read_csv(path + "constructors.csv")

#Filter races to modern era (2000–2024) + IDs
races_modern = races[(races["year"] >= 2000) & (races["year"] <= 2024)]
print("Modern races:", races_modern.shape)

modern_race_ids = races_modern["raceId"]

#Restrict results/qualifying to modern raceIds
results_modern = results[results["raceId"].isin(modern_race_ids)].copy()
qualifying_modern = qualifying[qualifying["raceId"].isin(modern_race_ids)].copy()

print("Results modern:", results_modern.shape)
print("Qualifying modern:", qualifying_modern.shape)

#Add year/name/circuitId to results via races_sub
races_sub = races[["raceId", "year", "name", "circuitId"]]
results_modern = results_modern.merge(races_sub, on="raceId", how="left")
# The 'year' column from races_sub is added directly as 'year', no need to handle duplicates.

#Restrict to 2000–2024 using the year column
mask = (results_modern["year"] >= 2000) & (results_modern["year"] <= 2024)
results_modern = results_modern[mask].copy()


qualifying_modern = qualifying_modern.merge(races_sub[["raceId", "year"]], on="raceId", how="left")
# The 'year' column from races_sub is added directly as 'year', no need to handle duplicates.
qualifying_modern = qualifying_modern[(qualifying_modern["year"] >= 2000) & (qualifying_modern["year"] <= 2024)].copy()


# Rename qualifying position to avoid confusion
q_sub = qualifying_modern[['raceId','driverId','position']] \
            .rename(columns={'position':'quali_pos'})

# Merge qualifying into results (final dataset)
df = results_modern.merge(
    q_sub,
    on=['raceId','driverId'],
    how='left'
)

print("Merged dataset shape:", df.shape)
print(df.head())

#ADVANCED CLEANING

#Merge df with status_df to get the status description
status_df = pd.read_csv(path + 'status.csv')

# Drop the 'status' column from df if it exists to avoid renaming conflicts during merge
if 'status' in df.columns:
    df = df.drop(columns=['status'])

df = df.merge(status_df, on='statusId', how='left')

#Apply the 'Finisher' filter
# This keeps only those who took the checkered flag (Finished or Lapped)
df = df[df["status"].astype(str).str.contains(r"Finished|\+\d+ Lap", regex=True, na=False)].copy()

#Final cleaning of ranks
# Ensure both positions are numeric and positive
# Check if 'positionOrder' exists, otherwise use 'finish_position'
if 'positionOrder' in df.columns:
    df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce')
elif 'finish_position' in df.columns:
    df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')

df["quali_pos"] = pd.to_numeric(df["quali_pos"], errors="coerce")

# Determine the current column names for position and grid for dropna and filtering
position_col_current = 'positionOrder' if 'positionOrder' in df.columns else 'finish_position'
grid_col_current = 'grid' if 'grid' in df.columns else 'grid_position'

df = df.dropna(subset=[position_col_current, 'quali_pos'])
df = df[(df[grid_col_current] > 0) & (df['quali_pos'] > 0)].copy()

print("After removing DNFs and invalid rows:", df.shape)

#Add driver names
drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
# Drop 'driver_name' from df if it already exists before merging
if 'driver_name' in df.columns:
    df = df.drop(columns=['driver_name'])
df = df.merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")

#Add team names
constructors = constructors.rename(columns={"name": "team_name"})
# Drop 'team_name' from df if it already exists before merging
if 'team_name' in df.columns:
    df = df.drop(columns=['team_name'])
df = df.merge(constructors[["constructorId", "team_name"]], on="constructorId", how="left")

#Rename key columns (only if they haven't been renamed already)
rename_cols = {}
if 'grid' in df.columns:
    rename_cols['grid'] = 'grid_position'
if 'positionOrder' in df.columns:
    rename_cols['positionOrder'] = 'finish_position'

if rename_cols:
    df = df.rename(columns=rename_cols)

print("Final cleaned dataset shape:", df.shape)
print(df.head())

#Era Classification
def assign_era(year):
    if 2000 <= year <= 2009: 
      return '2000-2009'
    elif 2010 <= year <= 2019: 
      return '2010-2019'
    else: return '2020-2024'

df['era'] = df['year'].apply(assign_era)

#Positions Gained/Lost
df['pos_change'] = df['grid_position'] - df['finish_position']

#Categorize Starting Position
df['grid_group'] = pd.cut(
    df['grid_position'], 
    bins=[0, 3, 10, 25], 
    labels=['Top 3', 'P4-P10', 'P11+']
)

#Binary Podium Indicator (1 for P1-P3, 0 otherwise)
df['is_podium'] = df['finish_position'].apply(lambda x: 1 if x <= 3 else 0)

#Quick validation
overall_corr, _ = stats.spearmanr(df["quali_pos"], df["finish_position"])
print(f"Verified Spearman Correlation (Qualifying vs Finish): {overall_corr:.4f}")

# Preview
print(df[[
    "year",
    "driver_name",
    "team_name",
    "grid_position",
    "quali_pos",
    "finish_position",
    "pos_change",
    "grid_group",
    "is_podium",
    "era"
]].head())
df.to_csv("data/cleaned_f1.csv", index=False)

print("Cleaned file saved successfully!")