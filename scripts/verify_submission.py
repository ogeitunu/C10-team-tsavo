import pandas as pd

def verify():
    sub_path = "submission.csv"
    print("🔍 Verifying Kaggle Submission Rules for submission.csv...")

    try:
        df = pd.read_csv(sub_path)
    except Exception as e:
        print(f"❌ Error reading submission file: {e}")
        return

    # 1. Check Columns
    expected_cols = ["QueryId", "DocumentId"]
    if list(df.columns) != expected_cols:
        print(f"❌ Invalid Columns! Found {list(df.columns)}, expected {expected_cols}")
    else:
        print("✔ Column Names Valid: QueryId, DocumentId")

    # 2. Check Row Count (200 test queries * 5 = 1,000 rows)
    if len(df) == 1000:
        print("✔ Total Row Count Valid: 1,000 rows")
    else:
        print(f"⚠️ Warning: Row count is {len(df)}, expected exactly 1,000 rows.")

    # 3. Check Exact 5 Rows per QueryId
    counts = df.groupby("QueryId").size()
    if (counts == 5).all() and len(counts) == 200:
        print("✔ Query Structure Valid: Exactly 5 ranked DocumentIds per test query")
    else:
        print("❌ Invalid Query Count Distribution!")

    # 4. Check Null Values
    if df.isnull().sum().sum() == 0:
        print("✔ No Null/NaN Values Found")
    else:
        print("❌ Submission contains Null/NaN values!")

    print("\n Verification complete! Ready to submit.")

if __name__ == "__main__":
    verify()
