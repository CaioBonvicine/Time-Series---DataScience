import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


ds = pd.read_csv(r"C:\Users\Meu-PC\Documents\C11ProjectTimeSeries\Time-Series---DataScience\dataset\energy_demand_hourly_brazil.csv", 
                delimiter=',')

ds['index'] = pd.to_datetime(ds['index'])
ds = ds.set_index('index')


energy = ds['hourly_demand']

plt.figure(figsize=(10,4))
energy.plot(title='Demanda Horaria de Energia no Brasil', xlabel='Ano', ylabel='MW')
plt.show()


decomp = seasonal_decompose(energy, model='additive', period=8766)

decomp.plot()
plt.show()