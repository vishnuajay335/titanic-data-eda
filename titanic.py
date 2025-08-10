import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# 1. Generate summary statistics
print(df.describe(include='all'))

# 2. Histograms for numeric features
numeric_cols = df.select_dtypes(include='number').columns.tolist()
plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# 3. Boxplots for numeric features
plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# 4. Correlation matrix and heatmap
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 5. Pairplot for selected features
selected_features = ['Survived', 'Pclass', 'Age', 'Fare']
df_pairplot = df[selected_features].copy()
df_pairplot['Age'].fillna(df_pairplot['Age'].median(), inplace=True)
sns.pairplot(df_pairplot, hue='Survived')
plt.show()

# Data Cleaning
cleaned_df = df.copy()
cleaned_df['Age'].fillna(cleaned_df['Age'].median(), inplace=True)
cleaned_df['Fare'].fillna(cleaned_df['Fare'].median(), inplace=True)
cleaned_df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)
cleaned_df.to_csv('Titanic_Cleaned.csv', index=False)
