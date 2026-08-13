# Data

This project uses the **Formula 1 World Championship (1950–2024)** dataset.

## Getting the raw data

Download the following files and place them in this `data/` folder:

- `races.csv`
- `results.csv`
- `qualifying.csv`
- `status.csv`
- `drivers.csv`
- `constructors.csv`

Source: [Formula 1 World Championship (1950–2024) on Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
(or the equivalent [Ergast API](http://ergast.com/mrd/) CSV export — make sure it's updated through the season you need).

## Generating the cleaned dataset

Once the raw CSVs are in `data/`, run the cleaning script from the project root:

```bash
python src/cleaning.py
```

This reads the six raw files, filters to the 2000–2024 seasons, removes DNFs,
merges in driver/team/qualifying info, engineers the derived columns
(`era`, `grid_group`, `pos_change`, `is_podium`), and writes the result to:

```
data/cleaned_f1.csv
```

`cleaned_f1.csv` is the only file the Streamlit app (`app.py`) actually reads,
and it's small enough (~7,100 rows) to commit to the repo so the live demo
doesn't need the raw Kaggle files at all.
