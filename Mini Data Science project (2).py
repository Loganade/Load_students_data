# import necessary libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# set style for our visualizations 
plt.style.use('seabor-v0_8-whitegrid')
sns.set_palette("virdis")

# Data Acquisition 
print ("Data Acquisition")
url =
df = pd.read_csv(url)
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Data Cleaning and Preprocessing
print("\n2. Data Cleaning and Preprocessing")

# Checking for Missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check Data Types 
print ("\nData types:")
print(df.dtypes)

# Rename columns for easier handling
df.columns = [col.lower().replace('','') for col in df.columns]
print("\nRenamed columns:")
print(df.columns)

# Calculate the total score for each student
d['total_score'] = df['math_score'] + df['writing_score']
df['average_score'] = df['total_score] / 3

# Display basic statistics
print("\nBasic statistics:")
print(df.describe())

# Exploratory Data Analysis (EDA)
print ("\n3. Exploratory Data Analysis ")

# Check the distribution of categorical variables
print("\nGender distribution:")
print(df['gender'].value_counts())

print("\nRace/ethnicity distribution: ")
print(df['gender'].value_counts())

print("\nParental level of education distribution:")
print(df['parental level of education'].value_counts())

print("\nLunch type distribution:")
print(df['lunch'].value_counts())

print("\nTest preparation course distribution:")
print(df['test_preparation_course'].value_counts())

# Data Visualization
print("\n4. Data Visualization")

# Plot 1: Distribution of average scores
plt.figure(figsize=(10, 6))
sns.histplot(df['average_score'], kde=True)
plt.title('Distribution of Average Scores', fontsize=15)
plt.xlabel('Average Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Plot 2: Correlation heatmap of numerical features
plt.figure(figsize=(10, 8))
numeric_cols = ['math_score', 'reading_score', 'writing_score', 'total_score', 'average_score']
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Scores', fontsize=15)
plt.show()

# Plot 3: Compare scores by gender
plt.figure(figsize=(12, 6))
sns.boxplot(x='gender', y='average_score', data=df)
plt.title('Average Scores by Gender', fontsize=15)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.show()

# Plot 4: Compare scores by test preparation course
plt.figure(figsize=(12, 6))
sns.boxplot(x='test_preparation_course', y='average_score', data=df)
plt.title('Average Scores by Test Preparation Course', fontsize=15)
plt.xlabel('Test Preparation Course', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.show()

# Plot 5: Compare scores by parental education level
plt.figure(figsize=(14, 6))
sns.boxplot(x='parental_level_of_education', y='average_score', data=df)
plt.title('Average Scores by Parental Education Level', fontsize=15)
plt.xlabel('Parental Education Level', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 6: Subject scores by race/ethnicity
plt.figure(figsize=(15, 8))
subjects = ['math_score', 'reading_score', 'writing_score']
df_melted = pd.melt(df, id_vars=['race/ethnicity'], value_vars=subjects,
                   var_name='subject', value_name='score')
sns.barplot(x='race/ethnicity', y='score', hue='subject', data=df_melted)
plt.title('Subject Scores by Race/Ethnicity', fontsize=15)
plt.xlabel('Race/Ethnicity', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.legend(title='Subject')
plt.tight_layout()
plt.show()

# Plot 7: Pairplot of scores
plt.figure(figsize=(10, 8))
sns.pairplot(df[['math_score', 'reading_score', 'writing_score', 'average_score']])
plt.suptitle('Pairplot of Scores', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

#  Machine Learning Modeling
print("\n5. Machine Learning Modeling")

# Define features and target variable
X = df.drop(['math_score', 'reading_score', 'writing_score', 'total_score', 'average_score'], axis=1)
y = df['average_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define categorical and numerical features
categorical_features = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
numerical_features = []  # No numerical features in this case excluding the target variables

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
    ],
    remainder='passthrough'
)

# Model 1: Linear Regression
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# Evaluate Linear Regression model
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Model Results:")
print(f"Mean Squared Error: {lr_mse:.2f}")
print(f"Root Mean Squared Error: {lr_rmse:.2f}")
print(f"R² Score: {lr_r2:.2f}")

# Cross-validation for Linear Regression
lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {lr_cv_scores}")
print(f"Average cross-validation R² score: {np.mean(lr_cv_scores):.2f}")

# Model 2: Random Forest Regressor
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

# Evaluate Random Forest model
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Model Results:")
print(f"Mean Squared Error: {rf_mse:.2f}")
print(f"Root Mean Squared Error: {rf_rmse:.2f}")
print(f"R² Score: {rf_r2:.2f}")

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {rf_cv_scores}")
print(f"Average cross-validation R² score: {np.mean(rf_cv_scores):.2f}")

# Compare actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Random Forest: Actual vs Predicted Average Scores', fontsize=15)
plt.xlabel('Actual Average Score', fontsize=12)
plt.ylabel('Predicted Average Score', fontsize=12)
plt.tight_layout()
plt.show()

# Extract feature importances from Random Forest
rf_model = rf_pipeline.named_steps['regressor']
feature_names = (
    rf_pipeline.named_steps['preprocessor']
    .transformers_[0][1]
    .get_feature_names_out(categorical_features)
)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Plot feature importances
plt.figure(figsize=(12, 8))
indices = np.argsort(feature_importances)[::-1]
plt.barh(range(len(indices)), feature_importances[indices], align='center')
plt.yticks(range(len(indices)), feature_names[indices])
plt.title('Random Forest Feature Importances', fontsize=15)
plt.xlabel('Relative Importance', fontsize=12)
plt.tight_layout()
plt.show()

#  Conclusion
print("\n6. Conclusion and Findings")
print("Key findings from our analysis:")
print("1. There's a strong correlation between math, reading, and writing scores.")
print("2. Students who completed the test preparation course tend to perform better.")
print("3. Parental education level shows an impact on student performance.")
print("4. The Random Forest model outperformed Linear Regression in predicting student average scores.")
print("5. The most important features in predicting student performance are [based on feature importance plot].")

print("\nNext steps could include:")
print("1. More advanced feature engineering")
print("2. Hyperparameter tuning of the Random Forest model")
print("3. Trying other models like Gradient Boosting or Neural Networks")
print("4. Exploring additional external datasets to enhance predictions")











