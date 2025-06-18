import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = sns.load_dataset('titanic')  

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(titanic_df['age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title("Age Distribution")

plt.subplot(1, 2, 2)
sns.boxplot(x='survived', y='age', data=titanic_df, palette='pastel')
plt.title("Age vs Survival")
plt.show()