import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic.head())

# Basic information about the dataset
print(titanic.info())

# Summary statistics
print(titanic.describe(include='all'))

# Handle missing data
# For simplicity, we'll drop rows with missing values
titanic = titanic.dropna(subset=['age', 'embarked'])

# Create new feature: Family size (sibsp + parch + 1)
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# Visualize the impact of different factors on survival

# Impact of Age on Survival
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='age', hue='survived', multiple='stack', kde=True)
plt.title('Impact of Age on Survival')
plt.show()

# Impact of Class on Survival
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='pclass', hue='survived')
plt.title('Impact of Class on Survival')
plt.show()

# Impact of Family Size on Survival
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='family_size', hue='survived')
plt.title('Impact of Family Size on Survival')
plt.show()

# Impact of Embarked (Port of Embarkation) on Survival
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='embarked', hue='survived')
plt.title('Impact of Port of Embarkation on Survival')
plt.show()

# Cross-tabulation for more detailed insights
pd.crosstab(titanic['pclass'], titanic['survived'], normalize='index').plot(kind='bar', stacked=True)
plt.title('Survival Rate by Class')
plt.ylabel('Proportion')
plt.show()

pd.crosstab(titanic['family_size'], titanic['survived'], normalize='index').plot(kind='bar', stacked=True)
plt.title('Survival Rate by Family Size')
plt.ylabel('Proportion')
plt.show()