import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

################################################################################
def window(df, w=7):
    """
    Cria janelas de tamanho w da coluna 'venda' do DataFrame df.
    Retorna X (entradas) e y (alvo).
    """
    dfs = df['venda'].values  
    X, y = [], []

    for i in range(len(dfs) - w):
        X.append(dfs[i:i+w])
        y.append(dfs[i+w])

    return np.array(X), np.array(y)

################################################################################
def train_model_01(df, sku_list):
    print(80*'-')
    print('Modeling')
    print(80*'-')

    for sku in sku_list:
        
        #######################################
        np.random.seed(42)
        print(40*'-')
        print(f'Training model for SKU: {sku}')
        print(40*'-')

        df_sku = df[df['sku'] == sku].copy()
        df_sku = df_sku.sort_values(by='data_venda') 
        print(f"Qtd. registros: {len(df_sku)}")

        X, y = window(df_sku, w=7)
        if len(X) == 0:
            print("Poucos dados para criar janelas. Pulando SKU.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        #######################################
        model_01 = Ridge()
        model_01.fit(X_train, y_train)

        y_pred_01 = model_01.predict(X_test)

        mse = mean_squared_error(y_test, y_pred_01)
        mae = mean_absolute_error(y_test, y_pred_01)

        print(f"MSE_01: {mse:.2f}")
        print(f"MAE_01: {mae:.2f}")

        test_dates = df_sku['data_venda'].iloc[-len(y_test):]

        #######################################
        entrada = df_sku['venda'].values[-7:].tolist()
        previsao_01 = []

        for _ in range(7):
            x_input = np.array(entrada[-7:]).reshape(1, -1)
            y_next = model_01.predict(x_input)[0]
            previsao_01.append(y_next)
            entrada.append(y_next)

        ultima_data = pd.to_datetime(df_sku['data_venda'].max())
        datas_futuras = [ultima_data + timedelta(days=i) for i in range(1, 8)]

        df_prev = pd.DataFrame({
            'data': datas_futuras,
            'sku': sku,
            'previsao': np.round(previsao_01, 1)
        })

        print(df_prev)

        df_prev.to_csv(f'doc/previsao_{sku}_Ridge_7_dias.csv', index=False)

        #######################################
        plt.figure(figsize=(10, 4))
        plt.plot(test_dates, y_test, label='Real')
        plt.plot(test_dates, y_pred_01, label='Previsto (teste)', linestyle='--')
        plt.plot(datas_futuras, previsao_01, label='Previsão (7 dias)', marker='o', linestyle='--', color='red')
        plt.title(f'Previsão vs Real - SKU {sku} - Modelo 01 Ridge')
        plt.xlabel('Data')
        plt.ylabel('Vendas')
        plt.legend()
        plt.grid(True)

        texto_metricas = f"MSE: {mse:.2f}\nMAE: {mae:.2f}"
        plt.text(
            0.02, 0.95, texto_metricas,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", edgecolor="gray")
        )

        plt.tight_layout()
        plt.savefig(f'img/previsao_vs_real_{sku}_Ridge.png')
        plt.close()
