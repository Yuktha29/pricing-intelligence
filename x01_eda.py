import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('retail_price.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nBasic Stats:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Clean data
df.dropna(inplace=True)

# Price distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['unit_price'], bins=40, color='steelblue')
plt.title('Unit Price Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['comp_1'], bins=40, color='coral')
plt.title('Competitor Price Distribution')
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150)
plt.close()
print("Saved: price_distribution.png")

# Price difference vs competitor
df['price_diff_comp1'] = df['unit_price'] - df['comp_1']
df['price_diff_comp2'] = df['unit_price'] - df['comp_2']
df['price_diff_comp3'] = df['unit_price'] - df['comp_3']

print("\nAvg price diff vs Competitor 1:", round(df['price_diff_comp1'].mean(), 2))
print("Avg price diff vs Competitor 2:", round(df['price_diff_comp2'].mean(), 2))
print("Avg price diff vs Competitor 3:", round(df['price_diff_comp3'].mean(), 2))

# Correlation heatmap
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=np.number).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# Sales by product category
plt.figure(figsize=(10, 5))
df.groupby('product_category_name')['total_price'].mean().sort_values(ascending=False).head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 Categories by Avg Revenue')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('category_revenue.png', dpi=150)
plt.close()
print("Saved: category_revenue.png")

# Save cleaned data
df.to_csv('retail_price_clean.csv', index=False)
print("\nCleaned data saved!")
print("Total records:", len(df))