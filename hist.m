% Load the CSV file into a table
data = readtable('Preprocessed_student_data.csv');

% Replace 'your_column' with the actual column name you want to plot
column_data = data{:,'UnemploymentRate'};

% Create the histogram
histogram(column_data, 'BinWidth', 1); % Adjust 'BinWidth' as needed
column_median = median(column_data)
% Add titles and labels
title("Histogram distribution of student's unemployment rate % ");
xlabel('Unemploymenr rate in %');
ylabel('Frequency');

% Show the plot
grid on;
e 