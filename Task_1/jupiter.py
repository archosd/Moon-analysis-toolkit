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
