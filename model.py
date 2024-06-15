import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump

# Load data
data = pd.read_csv('Deepression.csv')
data.dropna(inplace=True)

# Features and labels
X = data.drop('Depression State', axis=1)  # Replace 'Depression State' with your actual label column name
y = data['Depression State']

# Encode categorical features if necessary
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values
    ('scaler', StandardScaler()),  # Feature scaling
    ('model', RandomForestClassifier(random_state=42))  # Model
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Grid search for best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Save the best model
dump(best_model, 'mental_health_model.pkl')

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# rf_model = RandomForestClassifier(random_state=42)
# gb_model = GradientBoostingClassifier(random_state=42)

# # Train Random Forest
# rf_model.fit(X_train, y_train)
# rf_pred = rf_model.predict(X_test)
# rf_accuracy = accuracy_score(y_test, rf_pred)
# print(f'Random Forest Accuracy: {rf_accuracy}')

# # Train Gradient Boosting
# gb_model.fit(X_train, y_train)
# gb_pred = gb_model.predict(X_test)
# gb_accuracy = accuracy_score(y_test, gb_pred)
# print(f'Gradient Boosting Accuracy: {gb_accuracy}')
