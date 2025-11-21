import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Carregar os dados
ds = pd.read_csv(r"C:\Users\tobia\OneDrive\Documents\Time-Series---DataScience\dataset\energy_demand_hourly_brazil.csv", delimiter=',')

# 2. Arrumar o Índice de Tempo
# Convertendo a coluna 'index' para data real
ds['index'] = pd.to_datetime(ds['index'])
ds = ds.set_index('index')

# 3. Filtrar e TRANSFORMAR EM DIÁRIO (O passo mais importante!)
# Pegamos 2021 e 2022 e tiramos a média por dia.
# Isso remove o ruído horário e deixa o padrão anual claro.
energy = ds.loc["2021-01-01":"2022-12-31", "hourly_demand"]
energy_daily = energy.resample('D').mean() 

# Definir frequência explicitamente para evitar avisos (D = Daily)
energy_daily.index.freq = 'D'

# 4. Criar e Treinar o Modelo Holt-Winters
# seasonal_periods=365 -> O ciclo se repete a cada 365 dias
# damped_trend=True -> Impede que a previsão suba ou desça infinitamente (controla a tendência)
modelo = ExponentialSmoothing(
    energy_daily, 
    trend='add', 
    seasonal='add', 
    seasonal_periods=365,
    damped_trend=True
)

fit_modelo = modelo.fit()

# 5. Previsão para 2023 (365 dias)
forecast_2023 = fit_modelo.forecast(365)

# 6. Plotar o Gráfico Corrigido
plt.style.use("seaborn-v0_8")
plt.figure(figsize=(14, 6))

# Plotar histórico (média diária)
plt.plot(energy_daily.index, energy_daily, label='Histórico (2021-2022)', color='#4c72b0')

# Plotar previsão
plt.plot(forecast_2023.index, forecast_2023, label='Previsão 2023 (Holt-Winters)', color='#e67e22', linewidth=2.5)

plt.title("Previsão de Demanda de Energia (Média Diária) - 2023", fontsize=16, fontweight="bold")
plt.ylabel("MW", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Aplicamos a decomposição na variável 'energy_daily' que criamos antes.
# period=30: Usamos 30 para suavizar e ver a tendência mensal (ciclos de ~30 dias).
# Se quiser ver o padrão semanal (dias úteis x fim de semana), mude para period=7.
decomp = seasonal_decompose(energy_daily, model='additive', period=30)

# Criar o gráfico
fig = decomp.plot()

# Ajustar o tamanho para não ficar tudo espremido (opcional, mas recomendado)
fig.set_size_inches(14, 10) 

plt.tight_layout()
plt.show()