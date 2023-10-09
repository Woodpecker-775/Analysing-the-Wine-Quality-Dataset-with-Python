import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the red wine dataset
red_wine_data = pd.read_csv(r'C:\Users\Shafeeq\Downloads\Compressed\wine+quality\winequality-red.csv', sep=';')

# Load the white wine dataset
white_wine_data = pd.read_csv(r'C:\Users\Shafeeq\Downloads\Compressed\wine+quality\winequality-white.csv', sep=';')







# Combine the red and white wine
wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# Now, you can proceed with data cleaning and preparation as mentioned in the previous response.



#second part follows cleaning and prepare3 the data

# Combine the red and white wine datas

wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# Checking for the missing values
missing_values = wine_data.isnull().sum()


# Remove duplicate rows from data
wine_data = wine_data.drop_duplicates()

# Check data types
wine_data.dtypes

#Convert data typesif necessary
wine_data['quality'] = wine_data['quality'].astype(int)

# convert other columns if needed


from sklearn.preprocessing import StandardScaler

# Scale the numeric features
scaler = StandardScaler()
numeric_features = wine_data.select_dtypes(include=[float]).columns
wine_data[numeric_features] = scaler.fit_transform(wine_data[numeric_features])


#wine_data['is_red'] = (wine_data['type'] == 'red').astype(int)

#wine_data = pd.get_dummies(wine_data, columns=['type'])

from sklearn.model_selection import train_test_split

X = wine_data.drop(columns=['quality'])
y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




wine_data.to_csv('cleaned_wine_data.csv', index=False)


print(wine_data.head())




# almost kind of done.................................................


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load  cleaned and prepared dataset
wine_data = pd.read_csv('cleaned_wine_data.csv')

# Summary statistics
summary_stats = wine_data.describe()

# Pairplot to visualize relationships between numeric features
#sns.pairplot(wine_data, hue='is_red', diag_kind='kde')
#plt.show()

# Correlation matrix
correlation_matrix = wine_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Boxplot to visualize the distribution of 'quality' by wine type
plt.figure(figsize=(10, 6))
#sns.boxplot(x='is_red', y='quality', data=wine_data)
plt.title("Distribution of Quality by Wine Type")
plt.show()

# Histograms for some key features
plt.figure(figsize=(12, 5))
features_to_plot = ['alcohol', 'sulphates']
for feature in features_to_plot:
    sns.histplot(wine_data[feature], kde=True, bins=30)
plt.show()



#part 4

plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=wine_data)
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()

#Correlation of features with wine quality
correlation_with_quality = wine_data.corr()['quality'].sort_values(ascending=False)

# Bar plot to visualize feature correlations as output
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_quality.values, y=correlation_with_quality.index)
plt.title("Correlation of Features with Wine Quality")
plt.xlabel("Correlation Coefficient")
plt.show()


from sklearn.ensemble import RandomForestRegressor

# Fit Random Forest Regressor to rank feature importance
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importance = model.feature_importances_
feature_names = X.columns

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top N important features
top_n = 10
print("Top", top_n, "Important Features:")
print(feature_importance_df.head(top_n))


top_n = 10
top_features = feature_importance_df.head(top_n)

# Plot the bar graph
plt.figure(figsize=(9, 5))
plt.bar(top_features['Feature'], top_features['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top {} Important Features'.format(top_n))
plt.xticks(rotation=45)
plt.show()