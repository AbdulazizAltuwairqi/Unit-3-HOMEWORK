"""
Day 19 Activity: Transformation Practice
Tasks:
1) Load the skewed spend feature from data/day19_transform.csv
2) Apply log1p, square-root, and Yeo-Johnson transformations
3) Compare distributions and summary statistics before and after
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PowerTransformer


def main():
    data_path = Path(__file__).resolve().parent / "data" / "day19_transform.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Place day19_transform.csv in the data/ folder."
        )

    df = pd.read_csv(data_path)
    if "spend" not in df.columns:
        raise ValueError("The dataset must contain a 'spend' column.")

    spend = pd.to_numeric(df["spend"], errors="coerce")
    valid = spend.dropna()
    if valid.empty:
        raise ValueError("The 'spend' column has no numeric values to transform.")
    if (valid < 0).any():
        raise ValueError("log1p and square-root require non-negative spend values.")

    results = pd.DataFrame(index=valid.index)
    results["spend_original"] = valid
    results["spend_log1p"] = np.log1p(valid)
    results["spend_sqrt"] = np.sqrt(valid)
    results["spend_yeo_johnson"] = PowerTransformer(
        method="yeo-johnson", standardize=False
    ).fit_transform(valid.to_frame()).ravel()

    print("Summary statistics (mean, standard deviation, skewness):")
    summary = results.agg(["mean", "std", "skew"]).T
    print(summary.round(3))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, column in zip(axes.flat, results.columns):
        sns.histplot(results[column], kde=True, ax=ax)
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel("Value")
    fig.suptitle("Spend distributions before and after transformation")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
