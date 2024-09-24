import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Ignorar advertencias
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set(style='whitegrid', palette='muted', color_codes=True)
# plt.style.use('seaborn')  # Línea eliminada para evitar errores

# Cargar el archivo CSV
icu_data = pd.read_csv('ICU.csv')

# Vista previa de los datos
print("=== Vista Previa de los Datos ===")
print(icu_data.head())

# Información general del dataset
print("\n=== Información General del Dataset ===")
print(icu_data.info())

# Descripción estadística de los datos
print("\n=== Descripción Estadística de los Datos ===")
print(icu_data.describe())

# Eliminar columnas innecesarias como 'Unnamed: 0' si es redundante
if 'Unnamed: 0' in icu_data.columns:
    icu_data = icu_data.drop(columns=['Unnamed: 0'])

# Comprobar valores nulos
print("\n=== Valores Nulos por Columna ===")
print(icu_data.isnull().sum())

# Opcional: eliminar o imputar valores nulos
icu_data = icu_data.dropna()  # O puedes usar imputación con medianas o promedios

# Guardar el dataset limpio para uso posterior (opcional)
icu_data.to_csv('ICU_cleaned.csv', index=False)
print("\nDataset limpio guardado como 'ICU_cleaned.csv'.")

# ------------------------------
# Visualizaciones
# ------------------------------

# 1. Distribución de la Edad
plt.figure(figsize=(10,6))
sns.histplot(icu_data['Age'], kde=True, color='skyblue', bins=30)
plt.title('Distribución de Edad', fontsize=16)
plt.xlabel('Edad', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.show()

# 2. Distribución de la Presión Arterial Sistólica
plt.figure(figsize=(10,6))
sns.histplot(icu_data['SysBP'], kde=True, color='salmon', bins=30)
plt.title('Distribución de Presión Arterial Sistólica', fontsize=16)
plt.xlabel('Presión Arterial Sistólica (SysBP)', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.show()

# 3. Distribución de Supervivencia
plt.figure(figsize=(8,6))
sns.countplot(x='Survive', data=icu_data, palette='Set2')
plt.title('Distribución de Supervivencia', fontsize=16)
plt.xlabel('Supervivencia', fontsize=14)
plt.ylabel('Conteo', fontsize=14)
plt.show()

# 4. Relación entre Edad y Supervivencia
plt.figure(figsize=(10,6))
sns.boxplot(x='Survive', y='Age', data=icu_data, palette='Set3')
plt.title('Relación entre Edad y Supervivencia', fontsize=16)
plt.xlabel('Supervivencia', fontsize=14)
plt.ylabel('Edad', fontsize=14)
plt.show()

# 5. Relación entre Presión Arterial Sistólica y Supervivencia
plt.figure(figsize=(10,6))
sns.boxplot(x='Survive', y='SysBP', data=icu_data, palette='Set1')
plt.title('Relación entre Presión Arterial Sistólica y Supervivencia', fontsize=16)
plt.xlabel('Supervivencia', fontsize=14)
plt.ylabel('Presión Arterial Sistólica (SysBP)', fontsize=14)
plt.show()

# 6. Matriz de Correlación
plt.figure(figsize=(12,10))
correlation_matrix = icu_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Matriz de Correlación', fontsize=18)
plt.show()

# 7. Diagramas de Violín
plt.figure(figsize=(10,6))
sns.violinplot(x='Survive', y='Age', data=icu_data, palette='Pastel1')
plt.title('Distribución de Edad por Supervivencia', fontsize=16)
plt.xlabel('Supervivencia', fontsize=14)
plt.ylabel('Edad', fontsize=14)
plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(x='Survive', y='SysBP', data=icu_data, palette='Pastel2')
plt.title('Distribución de Presión Arterial Sistólica por Supervivencia', fontsize=16)
plt.xlabel('Supervivencia', fontsize=14)
plt.ylabel('Presión Arterial Sistólica (SysBP)', fontsize=14)
plt.show()

# 8. Gráfico de Pares (Pair Plot)
sns.pairplot(icu_data, hue='Survive', palette='bright')
plt.suptitle('Gráfico de Pares de Variables', y=1.02, fontsize=16)
plt.show()

# 9. Análisis de Valores Atípicos usando Boxplots para Múltiples Variables
numerical_features = ['Age', 'SysBP', 'HeartRate', 'RespRate', 'Temp']  # Ajusta según tus columnas
plt.figure(figsize=(15,10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=icu_data[feature], color='lightblue')
    plt.title(f'Detección de Outliers en {feature}', fontsize=14)
plt.tight_layout()
plt.show()

# 10. Visualización Interactiva con Plotly: Distribución de Edad
fig = px.histogram(icu_data, x='Age', nbins=30, title='Distribución de Edad', 
                   labels={'Age': 'Edad'}, opacity=0.7, color_discrete_sequence=['indianred'])
fig.update_layout(bargap=0.2)
fig.show()

# 11. Visualización Interactiva con Plotly: Presión Arterial vs Edad
fig = px.scatter(icu_data, x='Age', y='SysBP', color='Survive',
                 title='Presión Arterial Sistólica vs Edad',
                 labels={'Age': 'Edad', 'SysBP': 'Presión Arterial Sistólica (SysBP)', 'Survive': 'Supervive'},
                 hover_data=['HeartRate', 'RespRate', 'Temp'])
fig.update_traces(marker=dict(size=10,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()

# 12. Gráfico de Barras para Variables Categóricas Adicionales (Ejemplo: Gender)
if 'Gender' in icu_data.columns:
    plt.figure(figsize=(8,6))
    sns.countplot(x='Gender', hue='Survive', data=icu_data, palette='Set1')
    plt.title('Distribución de Género por Supervivencia', fontsize=16)
    plt.xlabel('Género', fontsize=14)
    plt.ylabel('Conteo', fontsize=14)
    plt.legend(title='Supervive')
    plt.show()

# 13. Mapa de Calor de la Distribución de Variables Categóricas
if 'Gender' in icu_data.columns:
    gender_survive = pd.crosstab(icu_data['Gender'], icu_data['Survive'])
    sns.heatmap(gender_survive, annot=True, fmt="d", cmap='YlGnBu')
    plt.title('Supervivencia por Género', fontsize=16)
    plt.xlabel('Supervive', fontsize=14)
    plt.ylabel('Género', fontsize=14)
    plt.show()

# 14. Análisis de Correlación Avanzada: Heatmap con Anotaciones de p-valores
import scipy.stats as stats

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    pvalues = pd.DataFrame(columns=df.columns, index=df.columns)
    for row in df.columns:
        for col in df.columns:
            if row == col:
                pvalues.loc[row, col] = np.nan
            else:
                _, pval = stats.pearsonr(df[row], df[col])
                pvalues.loc[row, col] = round(pval, 4)
    return pvalues

pvals = calculate_pvalues(icu_data)

plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f",
            cbar_kws={"label": "Coeficiente de Correlación"})
plt.title('Matriz de Correlación con Valores P', fontsize=18)

# Mostrar los valores p en el heatmap
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        if i != j:
            plt.text(j + 0.5, i + 0.5, f"p={pvals.iloc[i, j]}", 
                     horizontalalignment='center', verticalalignment='center', color='black', fontsize=8)

plt.show()

# 15. Análisis de la Distribución Temporal (Ejemplo: AdmissionDate)
if 'AdmissionDate' in icu_data.columns:
    icu_data['AdmissionDate'] = pd.to_datetime(icu_data['AdmissionDate'])
    icu_data.set_index('AdmissionDate', inplace=True)
    icu_data.resample('M').size().plot(kind='line', figsize=(12,6), title='Ingresos por Mes')
    plt.xlabel('Fecha de Admisión', fontsize=14)
    plt.ylabel('Número de Ingresos', fontsize=14)
    plt.show()
    # Restablecer el índice si necesitas seguir trabajando con otras columnas
    icu_data.reset_index(inplace=True)

# 16. Gráfico de Barras Apiladas para Comparar Múltiples Variables Categóricas (Ejemplo: Gender y Ethnicity)
if 'Gender' in icu_data.columns and 'Ethnicity' in icu_data.columns:
    stacked_data = pd.crosstab(icu_data['Gender'], icu_data['Ethnicity'])
    stacked_data.plot(kind='bar', stacked=True, figsize=(10,7), colormap='viridis')
    plt.title('Distribución de Etnicidad por Género', fontsize=16)
    plt.xlabel('Género', fontsize=14)
    plt.ylabel('Conteo', fontsize=14)
    plt.legend(title='Etnicidad')
    plt.show()

# 17. Gráfico de Mapa de Calor para Ver Interacciones entre Múltiples Variables
plt.figure(figsize=(14,12))
sns.heatmap(icu_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Matriz de Correlación Detallada', fontsize=18)
plt.show()

# 18. Gráfico de Sankey para Visualizar Flujos entre Categorías (Ejemplo: Género a Supervivencia)
if 'Gender' in icu_data.columns:
    gender_survive = pd.crosstab(icu_data['Gender'], icu_data['Survive'])
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Male", "Female", "Supervive", "No Supervive"],
            color=["lightblue", "lightpink", "green", "red"]
        ),
        link=dict(
            source=[0, 0, 1, 1],  # Indices corresponden a labels
            target=[2, 3, 2, 3],
            value=[gender_survive.loc['Male', '1'], gender_survive.loc['Male', '0'],
                   gender_survive.loc['Female', '1'], gender_survive.loc['Female', '0']]
        ))])

    fig.update_layout(title_text="Flujo de Supervivencia por Género", font_size=14)
    fig.show()
