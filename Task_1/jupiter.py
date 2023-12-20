# jupiter.py
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Moons:
	def __init__(self, database_name='jupiter.db'):
		data_folder = "data"
		database_path = os.path.join(data_folder,database_name)
		self.database_name = database_name
		self._load_data()

	def _load_data(self):
		# Load data from the SQLite database into a pandas DataFrame
		conn = sqlite3.connect(self.database_name)
		query = "SELECT * FROM moons"
		self.data = pd.read_sql_query(query, conn)
		conn.close()

	def get_summary_statistics(self):
	        # Calculate summary statistics for the dataset
		return self.data.describe()

	def get_correlation(self):
		# Calculate correlations between variables in the dataset
		return self.data.corr()

	def plot_data(self, x, y):
		# Plotting function to visualize the dataset
		plt.figure(figsize=(8, 6))
		sns.scatterplot(data=self.data, x=x, y=y)
		plt.title(f"Scatter plot of {y} against {x}")
		plt.xlabel(x)
		plt.ylabel(y)
		plt.show()

	def get_moon_data(self, moon_name):
		# Extract data for a specific moon
		moon_data = self.data[self.data['name'] == moon_name]
		return moon_data if not moon_data.empty else None

	def list_moons(self):
		# Return a list of available moon names
		return self.data['name'].tolist()

