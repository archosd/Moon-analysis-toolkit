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
		conn = "sqlite:///data/jupiter.db"
		query = "SELECT * FROM moons"
		self.data = pd.read_sql(query, conn)
