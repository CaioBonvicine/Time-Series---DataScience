import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

ds = pd.read_csv(r"C:\Users\Meu-PC\Documents\C11ProjectTimeSeries\Time-Series---DataScience\dataset\energy_demand_hourly_brazil.csv", delimiter=',')

ds['index'] = pd.to_datetime(ds['index'])
ds = ds.set_index('index')


energy = ds.loc["2021-01-01":"2022-12-31", "hourly_demand"]
energy_daily = energy.resample('D').mean() 

energy_daily.index.freq = 'D'

modelo = ExponentialSmoothing(
    energy_daily, 
    trend='add', 
    seasonal='add', 
    seasonal_periods=365,
    damped_trend=True
)

fit_modelo = modelo.fit()
forecast_2023 = fit_modelo.forecast(365)


plt.style.use("seaborn-v0_8")
plt.figure(figsize=(14, 6))


plt.plot(energy_daily.index, energy_daily, label='Histórico (2021-2022)', color='#4c72b0')


plt.plot(forecast_2023.index, forecast_2023, label='Previsão 2023 (Holt-Winters)', color='#e67e22', linewidth=2.5)

plt.title("Previsão de Demanda de Energia (Média Diária) - 2023", fontsize=16, fontweight="bold")
plt.ylabel("MW", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)


decomp = seasonal_decompose(energy_daily, model='additive', period=30)

fig = decomp.plot()

fig.set_size_inches(14, 10) 

plt.tight_layout()
plt.show()