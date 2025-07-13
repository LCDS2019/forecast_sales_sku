import datetime
import os

start = datetime.datetime.now()
os.system('clear')

################################################################################
# import libraries

import pg01_dataprep as data_prep
import pg02_ridge as ml
import pg03_arima as arima

################################################################################
# reading csv file

df = data_prep.read_csv_file('data/vendas.csv')
print(df.head(10))

################################################################################
# Goup by 'sku' and 'data_venda'

grouped_df = data_prep.group_by_sku(df)
print(grouped_df.head(20))

################################################################################
# Top4 sku for model training

grouped = data_prep.group_by_sku(df)
print(grouped.head(10))
sku_list = grouped.sort_values(by='qtd_points', ascending=False).head(5)['sku'].tolist()

print('\n')
print(f'Top 5 SKUs: {sku_list}')

################################################################################
# model training

ml.train_model_01(df, sku_list)
arima.train_model_arima(df, sku_list)


################################################################################

end = datetime.datetime.now()
time = end - start

hour = str(time.seconds // 3600).zfill(2)
min = str((time.seconds % 3600) // 60).zfill(2)
sec = str(time.seconds % 60).zfill(2)

msg_time = f'Time:{hour}:{min}:{sec} '
print('\n',msg_time)

################################################################################
