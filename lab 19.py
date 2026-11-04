"""
Day 19 Activity: Transformation Practice
Tasks:
1) Load skewed feature
2) Apply log1p, sqrt, and Yeo-Johnson
3) Compare before/after
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer

# TODO: Load data from data/day19_transform.csv
df = pd.read_csv("day19_transform.csv")
print(df)
# TODO: Apply transforms and compare summary stats
df["spend_log1p"] = np.log1p(df["spend"])
df["spend_sqrt"] = np.sqrt(df["spend"])
print(df)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df["spend_log1p"], ax=axes[0]); axes[0].set_title("Spend log1p")
sns.histplot(df["spend_sqrt"], ax=axes[1]); axes[1].set_title("Spend sqrt")
plt.show()