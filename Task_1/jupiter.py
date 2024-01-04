import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Moons:
	def __init__(self, database_name, database_path):
		self.database_name = database_name
		self.database_path = database_path

	def load_all_data(self):
		# Load data from the SQLite database into a pandas DataFrame
		conn = f"sqlite:///{self.database_path}{self.database_name}"
		query = "SELECT * FROM moons"
		return pd.read_sql(query, conn)

	def individual(self, name="Adrastea"):
		self.name = name
		conn = f"sqlite:///{self.database_path}{self.database_name}"
		query = "SELECT * FROM moons WHERE moon = ?"  # Must use a parameterized query for some reason
		return pd.read_sql_query(query, conn, params=(name,))
	def summary(self):
        	return self.data.describe()

    	def column_names(self, columns = 0):
		return self.data.head(columns)

	def correlation(self, variable_1 = "", variable_2 = "", correlation = 0):
        	self.variable_1 = variable_1 
        	self.variable_2 = variable_2 
        	correlation = self.data[variable_1].corr(self.data[variable_2])
        	return correlation

    	def detect_missing_values(self):
        """
        Detects missing values in each column of the DataFrame.

        Returns:
        - DataFrame showing count and percentage of missing values for each column
        """
        	missing_count = self.data.isnull().sum()
        	missing_percentage = (missing_count / len(self.data)) * 100
        	missing_info = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
        	return missing_info

    	def plot_histogram(self, column_name):
        	plt.figure(figsize=(8, 6))
	        plt.hist(self.data[column_name].dropna(), bins=10, color='hotpink', edgecolor='black')
        	plt.title(f'Histogram of {column_name}')
        	plt.xlabel(column_name)
        	plt.ylabel('Frequency')
        	plt.grid(True)
        	plt.show()

    	def plot_scatter(self, x_column, y_column, log_x=False, log_y=False):
        	plt.figure(figsize=(8, 6))

        	if log_x:
    			plt.scatter(self.data[x_column], self.data[y_column], marker='o')
            		plt.xscale('log')
        	elif log_y:
            		plt.scatter(self.data[x_column], self.data[y_column], marker='o')
            		plt.yscale('log')
        	else:
            		plt.scatter(self.data[x_column], self.data[y_column], marker='o')

        	plt.title(f'Scatter Plot: {x_column} vs {y_column}')
        	plt.xlabel(x_column)
        	plt.ylabel(y_column)
        	plt.grid(True)
        	plt.show()

    	def plot_with_regression(self, x_column, y_column, log_x=False, log_y=False):
        	plt.figure(figsize=(8, 6))

        	plt.scatter(self.data[x_column], self.data[y_column], marker='o')

        	if log_x:
            		plt.xscale('log')
        	if log_y:
            		plt.yscale('log')

        	x = self.data[x_column]
       	 	y = self.data[y_column]

        	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        	line = f'Linear Regression: y = {slope:.2f}x + {intercept:.2f}'

		plt.plot(x, slope * x + intercept, color='red', label=line)

        	plt.title(f'Scatter Plot with Linear Regression: {x_column} vs {y_column}')
        	plt.xlabel(x_column)
        	plt.ylabel(y_column)
        	plt.legend()
        	plt.grid(True)
        	plt.show()	
