# 🏎️ F1 Qualifying vs Race Outcomes

A data-driven analysis of whether qualifying position predicts race results in
Formula 1, covering the 2000–2024 seasons (7,134 classified race entries).
Includes a full cleaning/analysis pipeline and an interactive Streamlit
dashboard.

**Live demo:** _add your deployed Streamlit Cloud link here once deployed_

## Key findings

| Metric | Value |
|---|---|
| Spearman correlation (qualifying vs finish) | 0.785 |
| Pearson correlation | 0.782 |
| Mean Absolute Error | 3.06 positions |
| Podium probability, Top 3 grid | 73.9% |
| Podium probability, P4–P10 | 15.7% |
| Podium probability, P11+ | 1.9% |
| Chi-square (grid group vs podium) | χ² = 3017.1, p < 0.000001 |
| Linear regression R² | 0.61 |

Full write-up: [`F1_Project_Report.pdf`](./F1_Project_Report.pdf)

##Features

Statistical analysis pipeline — cleans and merges six raw F1 datasets (races, results, qualifying, drivers, constructors, status), removes DNFs, and engineers derived variables (era, grid_group, pos_change, is_podium).

Correlation analysis — Spearman and Pearson correlation between qualifying and finish position, computed overall and broken down by era (2000–2009, 2010–2019, 2020–2024).

Hypothesis testing — a chi-square test of independence checks whether starting grid group significantly predicts podium finishes.

Linear regression — a single-variable model (qualifying position → finish position) reports slope, intercept, and R², both as a standalone analysis and live inside the dashboard on any filtered subset of the data.
`
Interactive Streamlit dashboard — filter by year range, team, driver, and era; explore four tabs (scatter plot, position-change distribution, podium probability, and summary/era analysis) that all update dynamically.


## Project structure

```
.
├── app.py                  # Streamlit dashboard (entry point for the live demo)
├── src/
│   ├── cleaning.py         # Raw data -> data/cleaned_f1.csv
│   └── analysis.py         # Standalone statistical analysis + matplotlib plots
├── data/
│   ├── README.md           # Where to get the raw data
│   └── cleaned_f1.csv      # Cleaned dataset used by app.py (generate via cleaning.py)
├── .streamlit/
│   └── config.toml         # Dashboard theme
├── requirements.txt
└── README.md
```

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get the data (see data/README.md) and clean it
python src/cleaning.py

# 5. Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Saving this project locally

If you're starting from this workspace, download all the files, keep the
folder structure above, and initialize git:

```bash
cd f1-project
git init
git add .
git commit -m "Initial commit: F1 qualifying vs race outcomes project"
```

## Pushing to GitHub

```bash
# Create a new repo on github.com first (don't initialize it with a README),
# then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

## Deploying the live demo (Streamlit Community Cloud — free)

1. Make sure `data/cleaned_f1.csv` is committed to the repo (the `.gitignore`
   here is set up to allow it through even though other CSVs in `data/` are
   ignored).
2. Push your repo to GitHub (above).
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
4. Click **New app**, select your repo/branch, and set the main file path to
   `app.py`.
5. Click **Deploy**. Your app will be live at
   `https://<your-app-name>.streamlit.app` within a couple of minutes.
6. Paste that URL into the "Live demo" line at the top of this README.

Any time you push new commits to `main`, the live demo redeploys
automatically.

## Tech stack

Python · pandas · SciPy · scikit-learn · Streamlit · Plotly · Matplotlib · Seaborn

## Limitations

- DNFs and unclassified finishes are excluded, which biases the sample toward
  race completions.
- No contextual variables (weather, safety cars, tire strategy) are modeled.
- Grid group boundaries (Top 3 / P4–P10 / P11+) are a convention, not derived
  from the data.

See the full report for details.
