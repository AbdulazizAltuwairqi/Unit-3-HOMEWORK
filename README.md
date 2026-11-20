# Feature side of conflict exercise
This project loads spending data, applies transformations to reduce skewness, and compares the original and transformed distributions.

## What it does
- Loads data/day19_transform.csv. The CSV must contain a numeric spend column.
- Applies log1p, square-root, and Yeo-Johnson transformations.
- Prints the mean, standard deviation, and skewness for the original and transformed values.
- Plots histograms for comparison.

## Requirements
Install Python packages: pandas, numpy, matplotlib, seaborn, and scikit-learn.

## Run
Place day19_transform.csv in a data folder beside lab 19.py, then run:
python "lab 19.py"
The script reports a clear error if the CSV or required column is missing.
