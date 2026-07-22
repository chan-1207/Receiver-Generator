import pandas as pd

csv_path = "../building_register/building_register.csv"
df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")

target_cols = ["주구조", "주용도", "주지붕"]

for col in target_cols:
    print("\n====================")
    print(col)
    print("====================")
    print(df[col].fillna("NULL").value_counts().head(100))