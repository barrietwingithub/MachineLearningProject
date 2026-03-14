def engineer_features(df):

    df = df.copy()

    df['Spending_Income_Ratio'] = df['Total_Spending'] / df['Income']

    df['Transaction_Frequency'] = (
        df['Transaction_Count'] /
        df['Tenure_Days'].replace(0, 1)
    )

    drop_cols = [
        'Customer_ID',
        'First_Transaction_Date',
        'Last_Transaction_Date',
        'Total_Spending'
    ]

    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)

    return df
