import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Step 5: Create a Pipeline with the Preferred Parameters
preferred_params = {
    'pca__n_components': 6,  # Replace with your selected number of components
    'clf__max_depth': 15,      # Replace with your selected max depth
    'clf__min_samples_split': 2  # Replace with your selected min samples split
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),   # Scale the data
    ('pca', PCA(n_components=preferred_params['pca__n_components'])),  # Apply PCA
    ('clf', DecisionTreeClassifier(max_depth=preferred_params['clf__max_depth'], 
                                   min_samples_split=preferred_params['clf__min_samples_split']))  # Train a Decision Tree
])

# Step 6: Fit the Model with the Preferred Parameters
pipeline.fit(X_train, y_train)

# Step 7: Make Predictions on the Test Data
y_pred = pipeline.predict(X_test)

#8 Calculate and display accuracy, recall, precision, and F1-score
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1-score: {f1}")

# Step 9: Generate and Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

