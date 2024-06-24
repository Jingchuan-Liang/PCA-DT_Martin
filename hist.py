import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Preprocessed_student_data.csv'

# Load the CSV file with the correct delimiter
#data = pd.read_csv(file_path, delimiter=';')
df = pd.read_csv(file_path, encoding='utf-8')

# Replace 'your_column' with the actual column name you want to plot
column_data = df['Unemployment rate']

# Create a histogram of the data
plt.hist(column_data, bins=20, edgecolor='k')  # You can adjust the number of bins as needed

# Add titles and labels
plt.title("Histogram distribution of student's unemployment rate % ")
plt.xlabel('Unemploymenr rate in %')
plt.ylabel('Frequency')

# Show the plot
plt.show()