import pandas as pd

df = pd.read_csv("loan_approval_data.csv")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nApproval counts:\n", df["Loan_Status"].value_counts())
