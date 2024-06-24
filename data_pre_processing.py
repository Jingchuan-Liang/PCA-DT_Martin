import pandas as pd
import random

file_path = 'raw_student_data.csv'

# Load the CSV file with the correct delimiter
data = pd.read_csv(file_path, delimiter=';')

# Strip leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Print the columns of the DataFrame to debug
print("Columns in the DataFrame:", data.columns)

# Ensure 'GDP' column exists
gdp_column = "GDP"
unemployment = "Unemployment rate"

if gdp_column not in data.columns:
    raise ValueError(f"Column '{gdp_column}' not found in the DataFrame")

if unemployment not in data.columns:
    raise ValueError(f"Column '{unemployment}' not found in the DataFrame")

# Print descriptive statistics for the relevant columns
print(data[gdp_column].describe())
print(data[unemployment].describe())

# Calculate quantiles
q1 = data[gdp_column].quantile(0.33)
q2 = data[gdp_column].quantile(0.66)
q1_u = data[unemployment].quantile(0.33)
q2_u = data[unemployment].quantile(0.66)

print(f"33rd percentile (Q1): {q1}")
print(f"66th percentile (Q2): {q2}")

# Define function for classification
def classify_gdp(gdp):
    if gdp < q1:
        return '-1'
    elif gdp < q2:
        return '0'
    else:
        return '1'
    
def classify_un(un):
    if un < q1_u:
        return '-1'
    elif un < q2_u:
        return '0'
    else:
        return '1'

def classify_drop(ot):
    if ot == 'Dropout':
        return '1'
    else:
        return '0'
    
def classify_gra(ot):
    if ot == 'Graduate':
        return '1'
    else:
        return '0'

def classify_enroll(ot):
    if ot == 'Enrolled':
        return '1'
    else:
        return '0' 

def classify_random(ran):
    return random.randint(-1,1)

# Apply the classification function

data['Unemployment_rate_class'] = data[unemployment].apply(classify_un)
data['GDP_class'] = data[gdp_column].apply(classify_gdp)
data['Graduated_class'] = data['Output'].apply(classify_gra)
data['Dropout_class'] = data['Output'].apply(classify_drop)
data['Enrolled_class'] = data['Output'].apply(classify_enroll)
data['GDP_random'] = data[gdp_column].apply(classify_random)
data = data.drop(columns=['Inflation rate'])


data = data[data['Output'] != 'Enrolled']

#save to a new csv file
data.to_csv('Preprocessed_student_data.csv', index=False)