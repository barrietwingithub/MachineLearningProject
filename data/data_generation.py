import pandas as pd
import numpy as np


def create_sample_data(num_customers=1000, random_state=42):
    np.random.seed(random_state)

    customer_ids = [f'CUST_{i:04d}' for i in range(num_customers)]

    genders = np.random.choice(
        ['M', 'F', 'Male', 'Female', 'Unknown'],
        num_customers
    )

    ages = np.random.normal(45, 15, num_customers)
    ages = np.clip(ages, 18, 80).astype(int)

    incomes = np.random.lognormal(10, 0.5, num_customers)

    locations = np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], num_customers)

    demographics_df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Income': incomes,
        'Location': locations
    })

    num_transactions = int(num_customers * 1.5)

    transactions_df = pd.DataFrame({
        'Transaction_ID': [f'TXN_{i:06d}' for i in range(num_transactions)],
        'Customer_ID': np.random.choice(customer_ids, num_transactions),
        'Date': pd.date_range('2022-01-01', periods=num_transactions),
        'Amount': np.random.lognormal(4, 1.2, num_transactions),
        'Category': np.random.choice(
            ['Electronics', 'Clothing', 'Food', 'Books', 'Other'],
            num_transactions
        )
    })

    return demographics_df, transactions_df
