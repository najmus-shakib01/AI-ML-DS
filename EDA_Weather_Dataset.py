import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import missingno as msno
sns.set_theme(style="whitegrid")
plt.style.use('ggplot') 

try:
    df = pd.read_csv('https://raw.githubusercontent.com/alanjones2/dataviz/master/londonweather.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

print("\n=== Dataset Overview ===")
print(f"Shape of dataset: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

print("\n=== Dataset Information ===")
print(df.info())

print("\n=== Statistical Summary ===")
print(df.describe(include='all'))

print("\n=== Missing Values ===")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])  

plt.figure(figsize=(10, 6))
msno.matrix(df)
plt.title('Missing Values Visualization', fontsize=16)
plt.show()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(num_cols) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

print("\nMissing values after imputation:")
print(df.isnull().sum().sum(), "missing values remaining")

print("\n=== Univariate Analysis ===")

num_features = df.select_dtypes(include=['int64', 'float64']).columns
if len(num_features) > 0:
    plt.figure(figsize=(15, 10))
    cols = min(3, len(num_features))
    rows = (len(num_features) // cols) + (1 if len(num_features) % cols else 0)
    
    for i, col in enumerate(num_features):
        plt.subplot(rows, cols, i+1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

cat_features = df.select_dtypes(include=['object']).columns
if len(cat_features) > 0:
    plt.figure(figsize=(15, 10))
    cols = min(2, len(cat_features))
    rows = (len(cat_features) // cols) + (1 if len(cat_features) % cols else 0)
    
    for i, col in enumerate(cat_features):
        plt.subplot(rows, cols, i+1)
        sns.countplot(data=df, x=col)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("\n=== Bivariate Analysis ===")

if len(num_features) > 1:
    plt.figure(figsize=(12, 8))
    sns.pairplot(df[num_features])
    plt.suptitle('Pairplot of Numerical Features', y=1.02)
    plt.show()

if len(num_features) > 1:
    plt.figure(figsize=(10, 8))
    corr_matrix = df[num_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

if len(num_features) > 0 and len(cat_features) > 0:
    for num_col in num_features:
        for cat_col in cat_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=cat_col, y=num_col)
            plt.title(f'{num_col} vs {cat_col}')
            plt.xticks(rotation=45)
            plt.show()

print("\n=== Outlier Detection ===")
if len(num_features) > 0:
    plt.figure(figsize=(15, 10))
    cols = min(3, len(num_features))
    rows = (len(num_features) // cols) + (1 if len(num_features) % cols else 0)
    
    for i, col in enumerate(num_features):
        plt.subplot(rows, cols, i+1)
        sns.boxplot(data=df, y=col)
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

    for col in num_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

print("\n=== Encoding Categorical Variables ===")
if len(cat_features) > 0:
    label_encoder = LabelEncoder()
    for col in cat_features:
        df[col] = label_encoder.fit_transform(df[col])
        print(f"{col} encoded with {len(label_encoder.classes_)} classes")

if 'date' in df.columns or 'time' in df.columns:
    print("\n=== Time Series Analysis ===")
    date_col = 'date' if 'date' in df.columns else 'time'
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    for num_col in num_features:
        plt.figure(figsize=(12, 6))
        df[num_col].resample('M').mean().plot()
        plt.title(f'Monthly Average of {num_col}')
        plt.ylabel(num_col)
        plt.show()

if 'temperature' in df.columns and 'humidity' in df.columns:
    df['heat_index'] = df['temperature'] * df['humidity'] / 100
    print("\nAdded new feature: heat_index")

print("\n=== Final Dataset Preview ===")
print(df.head())

df.to_csv('cleaned_weather_data.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_weather_data.csv'")