import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class Moons:
	def __init__(self, database_name, database_path = ""):
		"""
		Initialize Moons class.

		Args:
		- database_name: Name of the database
		- database_path: Path to the database relative to working file.
		- data: Default value for data
		"""

		self.database_name = database_name
		self.database_path = database_path

		# Load data from the SQLite database into a pandas DataFrame
		conn = f"sqlite:///{self.database_path}{self.database_name}"
		query = "SELECT * FROM moons"
		self.data = pd.read_sql(query, conn)
	def all_data(self):
		"""
		Return the entire dataset.
		"""
		return self.data

	def individual(self, name="Adrastea"):
		"""
		Return data for a specific moon (default: Adrastea).

		Args:
		- name: Name of the moon
		"""
		self.name = name
		conn = f"sqlite:///{self.database_path}{self.database_name}"
		query = "SELECT * FROM moons WHERE moon = ?"  # Must use a parameterized query for some reason
		return pd.read_sql_query(query, conn, params=(name,))

	def group(self, group= "Galilean"):
		"""
		Return data for specific group of moons

		Args:
		-group: name of the group
		"""
		return self.data.loc[self.data["group"] == group]

	def summary(self):
		"""
		Return summary statistics of the dataset.
		"""
		return self.data.describe()

	def column_names(self, rows=0):
		"""
		Return the first 'rows' number of rows in the dataset.

		Args:
		- rows: Number of rows to return (default: 0), adjust if there are blank rows at the top of data etc.
		"""
		return self.data.head(rows)

	def correlation(self, variable_1="", variable_2="", correlation=0):
		"""
		Calculate correlation between two variables in the dataset.

		Args:
		- variable_1: Name of the first variable
		- variable_2: Name of the second variable
		- correlation: Placeholder for the calculated correlation (default: 0)
		"""
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
		"""
		Plot a histogram of the specified column.

		Args:
		- column_name: Name of the column to plot
		"""
		plt.figure(figsize=(8, 6))
		plt.hist(self.data[column_name].dropna(), bins=10, color='hotpink', edgecolor='black')
		plt.title(f'Histogram of {column_name}')
		plt.xlabel(column_name)
		plt.ylabel('Frequency')
		plt.grid(True)
		plt.show()

	def plot_scatter(self, x_column, y_column, log_x=False, log_y=False):
		"""
		Plot a scatter plot between x_column and y_column.

		Args:
		- x_column: Name of the column for x-axis
		- y_column: Name of the column for y-axis
		- log_x: Use logarithmic scale for x-axis (default: False)
		- log_y: Use logarithmic scale for y-axis (default: False)
		"""
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
		"""
		Plot a scatter plot with linear regression line and optional logarithmic scaling.

		Args:
		- x_column: Name of the column for x-axis
		- y_column: Name of the column for y-axis
		- log_x: Use logarithmic scale for x-axis (default: False)
		- log_y: Use logarithmic scale for y-axis (default: False)
		"""
		plt.figure(figsize=(8, 6))

		x = self.data[x_column]
		y = self.data[y_column]

		if log_x:
			x = np.log10(x)
		if log_y:
			y = np.log10(y)

		plt.scatter(x.dropna(), y.dropna(), marker='o')  # Drop NaN values before plotting

		x = x.dropna()
		y = y.dropna()

		slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
		line = f'Linear Regression: y = {slope:.2f}x + {intercept:.2f}'

		x_range = np.linspace(x.min(), x.max(), 100)
		regression_line = slope * x_range + intercept

		plt.plot(x_range, regression_line, color='red', label=line)

		plt.title(f'Scatter Plot with Linear Regression: {x_column} vs {y_column}')
		plt.xlabel(x_column)
		plt.ylabel(y_column)
		plt.legend()
		plt.grid(True)
		plt.show()

	def convert_distance(self, original_suffix, column):
		if original_suffix =="km":
			new_column_name = column[:-2]+"m"
			self.data[new_column_name] = self.data[column] * 1000
		elif original_suffix =="Mm":
			new_column_name = column[:-2]+"m"
			self.data["distance_m"] = self.data[column] * 1000000
		else:
			return "This conversion must be done manually with pandas."

	def convert_time(self, original_suffix, column):
		if original_suffix =="years":
			new_column_name = column[:-5]+"s"
			self.data[new_column_name] = self.data[column] * 31556952
		elif original_suffix =="days":
			new_column_name = column[:-4]+"s"
			self.data[new_column_name] = self.data[column] * 86400
		elif original_suffix == "hours":
			new_column_name = column[:-5]+"s"
			self.data[new_column_name] = self.data[column] * 3600
		else:
			return "This conversion must be done manually with pandas."

	def gen_sma(self, sma_column, distance_column, ecc_column):
		self.data[sma_column] = self.data[distance_column] * 1/(1-self.data[ecc_column])

	def prepare_data(self, time_column, semi_major_axis):
		self.data['T_squared'] = self.data[time_column]
		self.data['a_cubed'] = self.data[semi_major_axis]
	def test_train(self,time_column = "T_squared", axis_column = "a_cubed"):
		x = self.data[time_column]
		y = self.data[axis_column]
		x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 42)

		self.model = LinearRegression()
		self.model.fit(x_train, y_train)

		test_score = self.model.score(x_test, y_test)
		print(f"R-squared value on the test set: {test_score}")
