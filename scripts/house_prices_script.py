import pandas as pd

# Download House Price data
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

#Save locally
df.to_csv('data/house_prices.csv', index = False)

#Print number of houses, number of columns and top 5 rows.
print(f"Download {len(df)} houses")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())