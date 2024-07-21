import pandas as pd
from tabulate import tabulate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the CSV file
data = pd.read_csv('/content/drive/MyDrive/dataset_folder/mobile data/mobile phone price prediction.csv')

# Display the first few rows of the dataframe to understand its structure
print(tabulate(data.head(10), headers='keys', tablefmt='psql'))


# Convert the 'Price' column to a numerical type if needed
# If 'Price' column contains commas, uncomment the following line
data['Price'] = data['Price'].str.replace(',', '').astype(float)

data.drop(columns=['Unnamed: 0','Name'],inplace=True) # Drop columns that are not needed for the analysis
# Fill missing values with the mode of each column
for col in data.columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

from sklearn.preprocessing import LabelEncoder # Label encode categorical variables

le = LabelEncoder()
categorical_columns = ['Camera', 'Processor_name', 'Screen_resolution', 'Display', 'Battery', 'External_Memory']
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])



# Remove duplicate rows if any
data.drop_duplicates(inplace=True)
# Create dummy variables for categorical features
dummie_df = pd.get_dummies(data, drop_first=True)

# Identify features with a correlation greater than 0.15 with the target variable
corr_threshold = 0.15
corr_matrix = dummie_df.corr()
high_corr_features = corr_matrix.index[abs(corr_matrix["Price"]) > corr_threshold].tolist()

# Drop features with low correlation
dummie_df = dummie_df[high_corr_features]

# Split the dataset into training and testing sets
X = dummie_df.drop(columns='Price')
y = dummie_df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Initialize and train the Decision Tree model
tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(X_train, y_train)

# Predict with both models
y1_pred = linear_model.predict(X_test)
y2_pred = tree_model.predict(X_test)

# Calculate performance metrics
rmse_1 = mean_squared_error(y_test, y1_pred, squared=False)
r2_1 = r2_score(y_test, y1_pred)
rmse_2 = mean_squared_error(y_test, y2_pred, squared=False)
r2_2 = r2_score(y_test, y2_pred)

# Print performance metrics
print(f'Linear Regression RMSE: {rmse_1}')
print(f'Linear Regression R-Squared: {r2_1}')
print(f'Decision Tree RMSE: {rmse_2}')
print(f'Decision Tree R-Squared: {r2_2}')

# Plot actual vs predicted values for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y1_pred, label='Predicted', alpha=0.6)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Linear Regression)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Prediction Line')
plt.legend()
plt.show()

# Plot actual vs predicted values for Decision Tree
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y2_pred, label='Predicted', alpha=0.6)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Decision Tree)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Prediction Line')
plt.legend()
plt.show()

# Conclusion
if rmse_1 < rmse_2:
    print("Linear Regression has a lower RMSE and is better for this dataset.")
else:
    print("Decision Tree has a lower RMSE and is better for this dataset.")
