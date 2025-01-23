import pandas

def check_all_columns_numeric(df):
    """
    Check if all columns in a DataFrame are numeric. Return True if so, False otherwise.
    """
    # ----
    
    return df.select_dtypes(include=[int, float]).shape[1] == df.shape[1]


# unit test cases

import pandas as pd
df_numeric = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4.1, 5.2, 6.3]
})
print(check_all_columns_numeric(df_numeric))

df_mixed = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z']
})
print(check_all_columns_numeric(df_mixed))

df_empty = pd.DataFrame()
print(check_all_columns_numeric(df_empty))

df_varied_types = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [1.5, 2.5, 3.5],
    'C': [True, False, True]
})
print(check_all_columns_numeric(df_varied_types))