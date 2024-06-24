import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz

file_path = 'processed_data_integers.csv'

# Load the CSV file with the correct delimiter
#data = pd.read_csv(file_path, delimiter=';')
data = pd.read_csv(file_path, encoding='utf-8')

# Strip leading/trailing spaces from column names
#data.columns = data.columns.str.strip()

# Print the columns of the DataFrame to debug
print("Columns in the DataFrame:", data.columns)
print(data['Unemployment_rate_class'])

# Step 2: Preprocessing, x = data, y = label
X = data.drop(columns = ["Unemployment_rate_class","Output","Unemployment rate", "Mother's qualification", "Father's qualification","Mother's occupation","Father's occupation"])
y = data['Unemployment_rate_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=8)

# Step 5: Create a Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # Scale the data
    ('pca', PCA()),                 # Apply PCA
    ('clf', DecisionTreeClassifier())  # Train a Decision Tree
])

# Step 6: Hyperparameter Tuning using GridSearchCV
param_grid = {
    'pca__n_components': [4,5,6,7,8],  # Example values, adjust based on your dataset
    'clf__max_depth': [15,16,17,18,19,20,21,22,23],
    'clf__min_samples_split': [2, 5,7,9,10,12]
}

# Stratified K-Folds with shuffling
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")

# Step 7: Evaluate the Model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_score}")

# Step 8: Visualize the Decision Tree
# Fit the best model on the entire training data
best_model.fit(X_train, y_train)


# Generate feature names for the PCA components
n_components = best_model.named_steps['pca'].n_components_
pca_feature_names = [f'PCA_Component_{i}' for i in range(n_components)]

# Dynamically get class names
class_names = [str(class_name) for class_name in np.unique(y_train)]

# Visualize the Decision Tree using the correct feature names and class names
dot_data = export_graphviz(best_model.named_steps['clf'], out_file=None, 
                           feature_names=pca_feature_names, 
                           class_names=class_names,  # Dynamically determined class names
                           filled=True, rounded=True, 
                           special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("decision_tree")  # Save the decision tree visualization to a file
graph.view()  # Open the visualization in a viewer

# Step 9: Plotting Grid Search Results

# Extract results from grid search
results = grid_search.cv_results_

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Plot accuracy for different hyperparameters
plt.figure(figsize=(12, 6))

# Plotting PCA components vs. Mean Test Score
for depth in param_grid['clf__max_depth']:
    subset = results_df[results_df['param_clf__max_depth'] == depth]
    plt.plot(subset['param_pca__n_components'], subset['mean_test_score'], label=f'max_depth={depth}')

plt.xlabel('Number of PCA Components')
plt.ylabel('Mean Test Accuracy')
plt.title('Effect of PCA Components and Tree Depth on Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plotting max depth vs. Mean Test Score
plt.figure(figsize=(12, 6))

for n_components in param_grid['pca__n_components']:
    subset = results_df[results_df['param_pca__n_components'] == n_components]
    plt.plot(subset['param_clf__max_depth'], subset['mean_test_score'], label=f'n_components={n_components}')

plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Mean Test Accuracy')
plt.title('Effect of Tree Depth and PCA Components on Accuracy')
plt.legend()
plt.grid(True)
plt.show()