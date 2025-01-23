import pandas as pd

def non_numeric_columns_and_head(df, num_rows=20):
    """
    Identify non-numeric columns in a DataFrame and return their names and head.

    :param df: Pandas DataFrame to be examined.
    :param num_rows: Number of rows to include in the head (default is 20).
    :return: A tuple with two elements:
             1. List of column names that are not numeric (integer or float).
             2. DataFrame containing the head of the non-numeric columns.
    """
    # ----
    
    # Identify columns that are not of numeric data type
    non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    
    # Get the head of the non-numeric columns
    non_numeric_head = df[non_numeric_cols].head(num_rows).to_csv()
    
    return non_numeric_cols, non_numeric_head

# unit test cases
data = {
  'id': [1, 2, 3, 4, 5],
  'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
  'date_of_birth': pd.to_datetime(['1990-01-01', '1992-05-15', '1988-08-21', '1990-07-30', '1985-01-01']),
  'salary': [50000.00, 60000.00, 55000.00, 45000.00, 70000.00],
  'is_active': [True, False, True, False, True]
}
df = pd.DataFrame(data)
print(non_numeric_columns_and_head(df))

data = {
  'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
  'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
  'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com', 'eva@example.com']
}
df = pd.DataFrame(data)
print(non_numeric_columns_and_head(df))

data = {
  'id': [1, 2, 3],
  'name': ['Alice', 'Bob', 'Charlie'],
  'temperature': [98.6, 99.1, 98.7]
}
df = pd.DataFrame(data)
num_rows = 10
print(non_numeric_columns_and_head(df))