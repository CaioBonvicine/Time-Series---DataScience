import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


ds = pd.read_csv(r"C:\Users\Meu-PC\Documents\C11ProjectTimeSeries\Time-Series---DataScience\dataset\energy_demand_hourly_brazil.csv", 
                delimiter=',')

ds['index'] = pd.to_datetime(ds['index'])
ds = ds.set_index('index')


energy = ds.loc["2021-01-01":"2022-12-31", "hourly_demand"]

plt.style.use("seaborn-v0_8")
plt.figure(figsize=(14,5))
energy.plot(linewidth=1.2)
plt.title("Demanda Horária de Energia no Brasil (2021-2022)", fontsize=16, fontweight="bold")
plt.xlabel("Ano", fontsize=12)
plt.ylabel("MW", fontsize=12)
plt.tight_layout()
plt.show()


decomp = seasonal_decompose(energy, model='additive', period=2190)

decomp.plot()
plt.show()