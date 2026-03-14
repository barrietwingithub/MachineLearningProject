def integrate_data(demographics_df, transactions_df):

    customer_transactions = transactions_df.groupby('Customer_ID').agg({
        'Amount': ['sum', 'mean', 'count'],
        'Date': ['min', 'max']
    }).reset_index()

    customer_transactions.columns = [
        'Customer_ID',
        'Total_Spending',
        'Avg_Transaction_Value',
        'Transaction_Count',
        'First_Transaction_Date',
        'Last_Transaction_Date'
    ]

    customer_transactions['Tenure_Days'] = (
        customer_transactions['Last_Transaction_Date'] -
        customer_transactions['First_Transaction_Date']
    ).dt.days

    merged_df = demographics_df.merge(
        customer_transactions,
        on='Customer_ID',
        how='left'
    )

    return merged_df
