import pandas as pd

df = pd.read_parquet("/share/nas2_3/yhuang/_data/rgz/rgz_embedding_25.parquet")
print(df.columns)