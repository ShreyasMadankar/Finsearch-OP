import pandas as pd

# Load historical recession data
recession_data = pd.read_csv('recession_data.csv')

# Load asset performance data
asset_data = pd.read_csv('asset_data.csv')

# Data preprocessing
asset_data['Date'] = pd.to_datetime(asset_data['Date'])
asset_data.set_index('Date', inplace=True)
