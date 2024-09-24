import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
import scipy.stats as stats

# Ignorar advertencias
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set(style='whitegrid', palette='muted', color_codes=True)
# plt.style.use('seaborn')  # Línea eliminada para evitar errores

# Cargar el dataset limpio (si has ejecutado el script EDA y guardado 'ICU_cleaned.csv')



icu_data = pd.read_csv('ICU_cleaned.csv')




# 1. Seleccionar características y variable objetivo
# Excluir cualquier columna de ID si no se ha hecho anteriormente
# (ya manejado en el bloque anterior)
features = icu_data.drop('Survive', axis=1)
target = icu_data['Survive']

# 2. Identificar variables categóricas para codificar
categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()

# 3. Codificación de variables categóricas si existen
if categorical_features:
    features = pd.get_dummies(features, columns=categorical_features, drop_first=True)
    print(f"Variables categóricas codificadas: {categorical_features}")
else:
    print("No se encontraron variables categóricas para codificar.")

# 4. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=target)
print(f"Datos divididos en entrenamiento ({X_train.shape[0]}) y prueba ({X_test.shape[0]}).")

# 5. Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Entrenamiento de Modelos
# ------------------------------

# 1. Inicializar modelos
modelos = {
    'Regresión Logística': LogisticRegression(random_state=42),
    'Bosque Aleatorio': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Multilayer Perceptron (MLP)': MLPClassifier(random_state=42, max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine (SVM)': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 2. Entrenar modelos y almacenar resultados
resultados = {}
for nombre, modelo in modelos.items():
    if nombre in ['Regresión Logística', 'Multilayer Perceptron (MLP)']:
        # Modelos que requieren datos escalados
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        y_prob = modelo.predict_proba(X_test_scaled)[:,1]
    else:
        # Otros modelos que no requieren escalado
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        if hasattr(modelo, 'predict_proba'):
            y_prob = modelo.predict_proba(X_test)[:,1]
        else:
            # Para modelos que no tienen predict_proba (ej. algunos SVM), usar decision_function
            y_prob = modelo.decision_function(X_test)
            # Normalizar las probabilidades a [0,1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    
    # Calcular métricas
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Guardar resultados
    resultados[nombre] = {
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'AUC-ROC': roc_auc,
        'Predicciones': y_pred,
        'Probabilidades': y_prob
    }
    print(f"Modelo '{nombre}' entrenado y evaluado.")

# ------------------------------
# Evaluación de Modelos
# ------------------------------

# Crear un DataFrame para las métricas
metricas = []
for nombre, datos in resultados.items():
    cr = datos['Classification Report']
    metricas.append({
        'Modelo': nombre,
        'Precisión': cr['accuracy'],
        'Precisión (Clase 1)': cr['1']['precision'],
        'Recall (Clase 1)': cr['1']['recall'],
        'F1-Score (Clase 1)': cr['1']['f1-score'],
        'AUC-ROC': datos['AUC-ROC']
    })

metrics_df = pd.DataFrame(metricas)

print("\n=== Métricas de Rendimiento de los Modelos ===")
print(metrics_df)

# ------------------------------
# Visualización de Resultados de Modelos
# ------------------------------

# a. Matriz de Confusión para cada modelo
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {model_name}', fontsize=16)
    plt.xlabel('Predicción', fontsize=14)
    plt.ylabel('Realidad', fontsize=14)
    plt.show()

for nombre, datos in resultados.items():
    plot_confusion_matrix(datos['Confusion Matrix'], nombre)

# b. Curvas ROC para todos los modelos
plt.figure(figsize=(10,8))
for nombre, datos in resultados.items():
    fpr, tpr, _ = roc_curve(y_test, datos['Probabilidades'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.2f})')

# Línea diagonal
plt.plot([0,1], [0,1], 'k--')

plt.title('Curvas ROC de Todos los Modelos', fontsize=18)
plt.xlabel('Tasa de Falsos Positivos', fontsize=14)
plt.ylabel('Tasa de Verdaderos Positivos', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.show()

# c. Comparación de Métricas
metrics_melted = metrics_df.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')

plt.figure(figsize=(14,7))
sns.barplot(data=metrics_melted, x='Métrica', y='Valor', hue='Modelo')
plt.title('Comparación de Métricas de Rendimiento entre Modelos', fontsize=18)
plt.xlabel('Métrica', fontsize=14)
plt.ylabel('Valor', fontsize=14)
plt.ylim(0,1.05)
plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# d. Importancia de Características para modelos que la soportan
def plot_feature_importance(model, model_name, top_n=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train.columns
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(12,8))
        sns.barplot(x=feature_importances[:top_n], y=feature_importances.index[:top_n], palette='viridis')
        plt.title(f'Importancia de las Características - {model_name}', fontsize=16)
        plt.xlabel('Importancia', fontsize=14)
        plt.ylabel('Características', fontsize=14)
        plt.show()
    elif hasattr(model, 'coef_'):
        # Para modelos lineales como Regresión Logística
        coef = model.coef_[0]
        feature_names = X_train.columns
        coef_series = pd.Series(coef, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(12,8))
        sns.barplot(x=coef_series[:top_n], y=coef_series.index[:top_n], palette='coolwarm')
        plt.title(f'Coeficientes de las Características - {model_name}', fontsize=16)
        plt.xlabel('Coeficiente', fontsize=14)
        plt.ylabel('Características', fontsize=14)
        plt.show()

# Plot de importancia para Bosque Aleatorio y XGBoost
for nombre, datos in resultados.items():
    if nombre in ['Bosque Aleatorio', 'XGBoost']:
        modelo = modelos[nombre]
        plot_feature_importance(modelo, nombre)

# e. Visualización Interactiva con Plotly: Curvas ROC
fig = go.Figure()
for nombre, datos in resultados.items():
    fpr, tpr, _ = roc_curve(y_test, datos['Probabilidades'])
    roc_auc = auc(fpr, tpr)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{nombre} (AUC = {roc_auc:.2f})'))

# Línea diagonal
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Aleatorio', line=dict(dash='dash')))

# Configuración del layout
fig.update_layout(title='Curvas ROC Interactivas de Todos los Modelos', 
                  xaxis_title='Tasa de Falsos Positivos',
                  yaxis_title='Tasa de Verdaderos Positivos',
                  legend_title='Modelos',
                  template='plotly_white')

fig.show()

# f. Tabla de Reporte de Clasificación para todos los modelos
# Convertir los reportes de clasificación a DataFrame
def classification_report_to_df(cr, model_name):
    df = pd.DataFrame(cr).transpose()
    df['Modelo'] = model_name
    return df

report_dfs = []
for nombre, datos in resultados.items():
    cr = datos['Classification Report']
    report_df = classification_report_to_df(cr, nombre)
    report_dfs.append(report_df)

report_completo = pd.concat(report_dfs, axis=0)

print("\n=== Reporte de Clasificación Completo ===")
print(report_completo)

# Visualizar Precision, Recall y F1-Score para todos los modelos
report_melted = report_completo.reset_index().melt(id_vars=['Modelo', 'index'], 
                                                  value_vars=['precision', 'recall', 'f1-score'],
                                                  var_name='Métrica', value_name='Valor')

plt.figure(figsize=(16,8))
sns.barplot(data=report_melted, x='Métrica', y='Valor', hue='Modelo')
plt.title('Comparación de Precision, Recall y F1-Score entre Modelos', fontsize=18)
plt.xlabel('Métrica', fontsize=14)
plt.ylabel('Valor', fontsize=14)
plt.ylim(0,1.05)
plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

