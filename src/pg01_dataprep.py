import pandas as pd

################################################################################
# reading csv file

def read_csv_file(file):
    df = pd.read_csv(file, sep=',', encoding='utf-8')

    df['data_venda'] = pd.to_datetime(df['data_venda']).dt.date

    print(80*'-')
    print('DataFrame info:')
    print(80*'-')

    print(f'DataFrame shape:{df.shape} \n')
    #print(f'DataFrame columns: {df.columns.tolist()} \n')
    #print(f'DataFrame head:\n{df.head()}\n')
    #print(f'DataFrame describe:\n{df.describe()}\n')
    #print(f'DataFrame dtypes:\n{df.dtypes}\n')
    #print(f'DataFrame null values:\n{df.isnull().sum()}\n')

    df = df.sort_values(by=['sku', 'data_venda'])
    
    return df

################################################################################
# Goup by 'sku'

def group_by_sku(df):
    grouped_df = df.groupby('sku').agg(
        qtd_points=('sku', 'count'),
        data_min=('data_venda', 'min'),
        data_max=('data_venda', 'max')
    ).reset_index()

    grouped_df  = grouped_df.sort_values(by=['qtd_points'], ascending=False)    
    print(80*'-')
    print('Top4 sku selection:')
    print(80*'-')

    return grouped_df