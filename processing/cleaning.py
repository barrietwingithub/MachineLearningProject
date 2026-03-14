def clean_data(df):
    df = df.copy()
    gender_mapping = {
        'M': 'Male',
        'F': 'Female',
        'Male': 'Male',
        'Female': 'Female'
    }
    df['Gender'] = df['Gender'].map(gender_mapping).fillna('Unknown')
    numerical_cols = [
        'Age', 'Income',
        'Total_Spending',
        'Avg_Transaction_Value',
        'Transaction_Count',
        'Tenure_Days'
    ]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    threshold = df['Total_Spending'].quantile(0.75)
    df['High_Value_Customer'] = (df['Total_Spending'] > threshold).astype(int)
    return df
