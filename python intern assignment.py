import  pandas as pd
import numpy  as np
from sklearn.preprocessing  import StandardScaler
import  matplotlib.pyplot as plt
import  seaborn as sns

# importing the red wine dataset
red_winedata = pd.read_csv(r'D:\pythons\wine+quality\winequality-red.csv', sep=';')

# importing the white wine dataset
white_winedata = pd.read_csv(r'D:\pythons\wine+quality\winequality-red.csv', sep=';')

# Combining the red and white wine databases
winedata = pd.concat([red_winedata, white_winedata], ignore_index=True)

#  proceeding with data cleaning  and preparation as mentioned in the previous response.
#second part is cleaning and  preparing the data
# Combine the red  and white wine datas

winedata = pd.concat([red_winedata, white_winedata], ignore_index=True)

# Checking for the  values missing
missing_values =  winedata.isnull().sum()

# Remove duplicate rows from data
winedata = winedata.drop_duplicates()

# Check data types
winedata.dtypes

#Convert data typesif necessary
winedata['quality'] =  winedata['quality'].astype(int)

# convert other columns if needed

from  sklearn.preprocessing  import StandardScaler

# Scale the numeric features
scaler = StandardScaler()

numeric_features = winedata.select_dtypes(include=[float]).columns

winedata[numeric_features] =  scaler.fit_transform(winedata[numeric_features])


#winedata['is_red'] =  (winedata['type']  == 'red').astype(int)

#winedata  = pd.get_dummies(winedata, columns=['type'])

from sklearn.model_selection import train_test_split

X = winedata.drop(columns=['quality'])
y = winedata['quality']
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

winedata.to_csv('cleaned_winedata.csv',index=False)

print(winedata.head())


# almost kind of done.................................................

# Load  cleaned and prepared dataset
winedata = pd.read_csv('cleaned_winedata.csv')

# Summary statistics
summary_stats = winedata.describe()

# Pairplot to visualize relationships between numeric features
#sns.pairplot(winedata, hue='is_red', diag_kind='kde')
#plt.show()

# Correlation matrix
correlation_matrix = winedata.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Boxplot to visualize the distribution of 'quality' by wine type
plt.figure(figsize=(9, 5))
#sns.boxplot(x='is_red', y='quality', data=winedata)
plt.title("Distribution of Quality by Wine Type")
plt.show()

# Histograms for some key features
plt.figure(figsize=(12, 5))
features_to_plot = ['alcohol', 'sulphates']
for feature in features_to_plot:
    sns.histplot(winedata[feature], kde=True, bins=30)
plt.show()

#part 4

plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=winedata)
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()

#Correlation of features with wine quality
correlation_with_quality = winedata.corr()['quality'].sort_values(ascending=False)

# Bar plot to visualize feature correlations as output
plt.figure(figsize=(9, 5))
sns.barplot( x=correlation_with_quality.values , y=correlation_with_quality.index )
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
feature_importance_df =  pd.DataFrame({'Feature':  feature_names, 'Importance': feature_importance})
feature_importance_df  = feature_importance_df. sort_values(by= 'Importance', ascending=False )

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
