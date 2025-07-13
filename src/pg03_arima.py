import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

################################################################################
def train_model_arima(df, sku_list):
    print(80*'-')
    print('Modeling with ARIMA')
    print(80*'-')

    for sku in sku_list:
        
        #######################################
        print(40*'-')
        print(f'Training ARIMA model for SKU: {sku}')
        print(40*'-')

        df_sku = df[df['sku'] == sku].copy()
        df_sku = df_sku.sort_values(by='data_venda') 
        print(f"Qtd. registros: {len(df_sku)}")

        serie = df_sku['venda'].values

        if len(serie) < 20:
            print("Série muito curta para ARIMA. Pulando SKU.")
            continue

        #######################################

        n = len(serie)
        split = int(n * 0.8)
        train, test = serie[:split], serie[split:]

        #######################################
        try:
            model = ARIMA(train, order=(7,1,2))
            model_fit = model.fit()
        except:
            print("Erro ao ajustar ARIMA. Pulando SKU.")
            continue

        y_pred = model_fit.forecast(steps=len(test))
        mse_02 = mean_squared_error(test, y_pred)
        mae_02 = mean_absolute_error(test, y_pred)

        print(f"MSE_ARIMA: {mse_02:.2f}")
        print(f"MAE_ARIMA: {mae_02:.2f}")

        #######################################

        full_model = ARIMA(serie, order=(2,1,2)).fit()
        forecast = full_model.forecast(steps=7)

        ultima_data = pd.to_datetime(df_sku['data_venda'].max())
        datas_futuras = [ultima_data + timedelta(days=i) for i in range(1, 8)]

        df_prev = pd.DataFrame({
            'data': datas_futuras,
            'sku': sku,
            'previsao': np.round(forecast, 1)
        })

        print(df_prev)
        df_prev.to_csv(f'doc/previsao_{sku}_arima_7_dias.csv', index=False)

        #######################################
        test_dates = df_sku['data_venda'].iloc[split:]
        
        plt.figure(figsize=(10, 4))
        plt.plot(test_dates, test, label='Real')
        plt.plot(test_dates, y_pred, label='Previsto (teste)', linestyle='--')
        plt.plot(datas_futuras, forecast, label='Previsão (7 dias)', marker='o', linestyle='--', color='red')

        plt.title(f'Previsão ARIMA - SKU {sku}')
        plt.xlabel('Data')
        plt.ylabel('Vendas')
        plt.legend()
        plt.grid(True)

        # Métricas no gráfico
        texto_metricas = f"MSE: {mse_02:.2f}\nMAE: {mae_02:.2f}"
        plt.text(
            0.02, 0.95, texto_metricas,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", edgecolor="gray")
        )

        plt.tight_layout()
        plt.savefig(f'img/previsao_vs_real_{sku}_arima.png')
        plt.close()
