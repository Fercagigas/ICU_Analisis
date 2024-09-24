from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
icu_data = pd.read_csv('ICU.csv')

# Crear el objeto KaplanMeierFitter
kmf = KaplanMeierFitter()

# Tiempo de seguimiento (puedes usar la variable 'Age' como un proxy de tiempo si no tienes una variable de tiempo)
durations = icu_data['Age']  # Aquí deberías usar el tiempo de supervivencia si lo tienes
event_observed = icu_data['Survive']  # 1 = sobrevivió, 0 = no sobrevivió

# Ajustar el modelo
kmf.fit(durations, event_observed)

# Graficar la curva de Kaplan-Meier
kmf.plot_survival_function()
plt.title('Curva de Supervivencia Kaplan-Meier')
plt.xlabel('Edad')
plt.ylabel('Probabilidad de Supervivencia')
plt.show()
