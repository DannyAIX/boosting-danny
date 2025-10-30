# Proyecto con Boosting Algorithms
#  CRISP-DM CRISP-DM significa Cross Industry Standard Process for Data Mining,
# y es el estándar más usado en la industria para desarrollar proyectos de análisis

###############   0. Business Understanding
# Este conjunto de datos proviene originalmente del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales.
# El objetivo es predecir en base a medidas diagnósticas si un paciente tiene o no diabetes.

###############   1. Data Understanding

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
total_data = pd.read_csv("https://breathecode.herokuapp.com/asset/internal-link?id=930&path=diabetes.csv", sep=',')
total_data.to_csv("/workspaces/boosting-danny/data/raw/total_data.csv", sep=',', index=False)

# Exploración inicial
print("Exploración inicial")
print(total_data.head())
print("Filas y columnas:", total_data.shape)
print("Info:")
print(total_data.info())
print("Estadísticas Descriptivas (originales):")
print(total_data.describe())

# 2.1 Verificar valores nulos o ceros sospechosos
print("\nValores nulos por columna:")
print(total_data.isnull().sum())

# 2.2 Contar valores cero en variables que no deberían tenerlos
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    zeros = (total_data[col] == 0).sum()
    print(f"{col}: {zeros} valores cero")

# 2.3 Visualización rápida de distribuciones
total_data.hist(figsize=(10, 8))
plt.suptitle("Distribución de variables antes de limpieza")
plt.show()

###############   3. Data Preparation
interim_data = total_data.copy()

# Reemplazar ceros por NaN (ya hecho antes)
interim_data[cols_with_zeros] = interim_data[cols_with_zeros].replace(0, np.nan)

# Imputar valores faltantes con la mediana sin usar inplace en el slice
for col in cols_with_zeros:
    median_val = interim_data[col].median()
    interim_data[col] = interim_data[col].fillna(median_val)

# Verificar que ya no haya nulos
print("\nValores nulos después de la imputación:")
print(interim_data.isnull().sum())

# Estadísticas descriptivas después de la limpieza
print("\nEstadísticas Descriptivas (dataset limpio):")
print(interim_data.describe())

#No se va a escalar o normalizar ya que se utilizará el modelo de arboles de decision 

# Identificación de Outliers

# se utilizará IQR IQR significa Interquartile Range, o rango intercuartílico. 
# Es una medida de dispersión que se centra en la parte “central” de tus datos, ignorando valores extremos.

# Columnas numéricas a revisar
num_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Función para detectar outliers
def detect_outliers(df, cols):
    outlier_dict = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        outlier_dict[col] = outliers
        print(f"{col}: {len(outliers)} outliers")
    return outlier_dict

outliers = detect_outliers(interim_data, num_cols)

#Capear (winsorize): reemplazar valores extremos por percentiles.
#Eliminar: solo si hay muy pocos outliers.
#quí usamos capping al 1er y 99º percentil, robusto para árboles:

for col in num_cols:
    lower = interim_data[col].quantile(0.01)
    upper = interim_data[col].quantile(0.99)
    interim_data[col] = np.clip(interim_data[col], lower, upper)

# ------------Importancia de las variables

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Separar variables
X = interim_data.drop("Outcome", axis=1)
y = interim_data["Outcome"]

# Separar la data en train y test

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar árbol de decisión
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Predicción y evaluación
y_pred = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(importances)

# Visualización
importances.plot(kind="bar", figsize=(10,5), title="Feature Importance")
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_text
import joblib  # Para guardar el modelo

############### 4. Probar distintos criterios de pureza #################

criterios = ['gini', 'entropy', 'log_loss']
for crit in criterios:
    tree = DecisionTreeClassifier(random_state=42, criterion=crit)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print(f"\nCriterio: {crit}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Visualizar feature importance
    importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importance:")
    print(importances)
    importances.plot(kind="bar", figsize=(10,5), title=f"Feature Importance ({crit})")
    plt.show()

############### 5. Optimización de hiperparámetros #################

# Definir el grid de búsqueda
param_grid = {
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nMejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Evaluar modelo optimizado
best_tree = grid_search.best_estimator_
y_pred_best = best_tree.predict(X_test)
print("\nEvaluación del árbol optimizado:")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# Guardar el modelo optimizado
joblib.dump(best_tree, "decision_tree_optimized.pkl")
print("\nModelo guardado como 'decision_tree_optimized.pkl'")

# visualizar árbol en texto
tree_rules = export_text(best_tree, feature_names=list(X.columns))
print("\nReglas del árbol optimizado:")
print(tree_rules)

#-------------------------------------------------------------------------- RANDOM FOREST
#random forest es una agrupación de árboles generados con porciones aleatorias de los datos y con criterios también aleatorios. 
# Esta visión nos permitiría mejorar la efectividad del modelo cuando un árbol individual no es suficiente.
#En este proyecto se centrará en esta idea entrenando el conjunto de datos para mejorar el accuracy
#Random Forest suele superar a un solo árbol porque reduce la varianza al promediar muchos árboles.

############### 0. Importar librerías necesarias ###############
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

############### 1. Entrenar Random Forest ###############

# Definir el modelo base
rf = RandomForestClassifier(
    n_estimators=100,        # número de árboles, se puede aumentar a 200-500, más árboles generalmente mejora el accuracy 
    #pero sube el tiempo de entrenamiento
    max_depth=None,           # profundidad máxima de los árboles, limitarlo puede evitar overfitting.
    min_samples_split=2,      # mínimo de muestras para dividir un nodo
    min_samples_leaf=1,       # mínimo de muestras por hoja
    criterion='gini',         # criterio de pureza
    random_state=42,
    n_jobs=-1                 # usar todos los cores
)

# Entrenar
rf.fit(X_train, y_train)

# Predicciones
y_pred_rf = rf.predict(X_test)

# Evaluación
print("Accuracy Random Forest:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

############### 2. Importancia de las variables ###############
importances_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance Random Forest:")
print(importances_rf)

# Visualización
importances_rf.plot(kind="bar", figsize=(10,5), title="Feature Importance (Random Forest)")
plt.show()

############### 3. Probar variando hiperparámetros ###############
from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_rf.fit(X_train, y_train)

print("\nMejores hiperparámetros Random Forest:")
print(grid_search_rf.best_params_)

# Evaluar el mejor modelo
best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print("\nEvaluación del Random Forest optimizado:")
print("Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))

############### 4. Guardar el modelo ###############
joblib.dump(best_rf, "random_forest_optimized.pkl")
print("\nModelo Random Forest guardado como 'random_forest_optimized.pkl'")

#-------------------------------------------------------------------------- COMPARACIÓN DE MODELOS
from sklearn.metrics import precision_score, recall_score, f1_score

print("\n================= COMPARACIÓN FINAL ENTRE MODELOS =================")

# --- Métricas Árbol de Decisión Optimizado ---
acc_tree = accuracy_score(y_test, y_pred_best)
prec_tree = precision_score(y_test, y_pred_best)
rec_tree = recall_score(y_test, y_pred_best)
f1_tree = f1_score(y_test, y_pred_best)

# --- Métricas Random Forest Optimizado ---
acc_rf = accuracy_score(y_test, y_pred_best_rf)
prec_rf = precision_score(y_test, y_pred_best_rf)
rec_rf = recall_score(y_test, y_pred_best_rf)
f1_rf = f1_score(y_test, y_pred_best_rf)

# --- Mostrar comparativa numérica ---
print(f"\nAccuracy Árbol de Decisión: {acc_tree:.4f}")
print(f"Accuracy Random Forest:     {acc_rf:.4f}")
print(f"Mejora en accuracy:         {((acc_rf - acc_tree) * 100):.2f}%")

print("\n--- Métricas detalladas ---")
print(f"{'Métrica':<15}{'Árbol de Decisión':<20}{'Random Forest'}")
print(f"{'Accuracy':<15}{acc_tree:<20.4f}{acc_rf:.4f}")
print(f"{'Precision':<15}{prec_tree:<20.4f}{prec_rf:.4f}")
print(f"{'Recall':<15}{rec_tree:<20.4f}{rec_rf:.4f}")
print(f"{'F1-Score':<15}{f1_tree:<20.4f}{f1_rf:.4f}")

# --- Visualización de comparación ---
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
tree_scores = [acc_tree, prec_tree, rec_tree, f1_tree]
rf_scores = [acc_rf, prec_rf, rec_rf, f1_rf]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, tree_scores, width, label='Decision Tree', color='skyblue')
bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='orange')

ax.set_ylabel('Puntaje')
ax.set_title('Comparación de métricas entre Árbol de Decisión y Random Forest')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar in bars1 + bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# --- Conclusión rápida ---
if acc_rf > acc_tree:
    print(f"\n✅ El modelo Random Forest supera al Árbol de Decisión en un {((acc_rf - acc_tree)*100):.2f}% de accuracy promedio.")
else:
    print(f"\n⚠️ El Árbol de Decisión tuvo mejor desempeño que el Random Forest en este caso.")


#-----------------------------BOOSTING 
#Boosting es una composición de modelos (generalmente árboles de decisión) secuencial en la cual el modelo nuevo 
# persigue corregir los errores del anterior.

#Árbol de decisión → aprende de una sola estructura (alta varianza).
#Random Forest → promedia árboles entrenados en paralelo (reduce varianza).
#Boosting → entrena árboles de forma secuencial, donde cada nuevo árbol corrige los errores del anterior 
# (reduce sesgo y mejora precisión). Esto hace que Boosting (sobre todo Gradient Boosting, XGBoost, AdaBoost o LightGBM) 
# logre mejores resultados en datasets tabulares.

#XGBoost (eXtreme Gradient Boosting) es la implementación más eficiente del algoritmo de gradient boosting. 
# Se ha desarrollado buscando rapidez y precisión, y hasta ahora es la mejor implementación, 
# superando en tiempos de entrenamiento a la de sklearn. 
# La reducción de tiempos se debe a que proporciona métodos para paralelizar las tareas, 
# así como flexibilidad a la hora de entrenar el modelo y es más robusto, 
# pudiendo incluir mecanismos de poda de los árboles para ahorrar tiempos de procesamiento. 
# Siempre que la tengamos disponible, esta es la alternativa que se debería utilizar frente a la de sklearn.


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Modelo base
xgb = XGBClassifier(
    n_estimators=200,       # Cuantos más árboles, menor sesgo pero mayor riesgo de sobreajuste y tiempo de cómputo
                            #número de árboles 100 árboles suelen ser suficientes para datasets simples.
                            #500–1000 se usan en datasets grandes o muy no lineales.
                            #200 es un punto intermedio ideal para obtener buena performance sin sobreentrenar ni tardar mucho

    learning_rate=0.05,     # Controla cuánto peso se da a cada nuevo árbol en la corrección de errores.
                            #velocidad de aprendizaje (más bajo = más preciso pero más lento)
                            #Valores altos (≥0.1) → el modelo aprende rápido, pero puede sobreajustar.
                            #Valores bajos (≤0.03) → el modelo aprende más lento pero más fino.
                            #0.05 es un punto seguro que permite combinar estabilidad y precisión.
                            #En boosting, hay una regla empírica: “A menor learning_rate, mayor número de árboles necesitas”.

    max_depth=4,            # Profundidad de los árboles Determina la profundidad máxima de cada árbol base.
                            # Árboles más profundos → mayor capacidad de capturar interacciones, pero también más overfitting.
                            # En este dataset, las relaciones entre variables (como Glucose, BMI, Age, etc.) 
                            # no son extremadamente complejas. Profundidades entre 3 y 6 suelen ser óptimas.
                            #Con max_depth=4, el modelo mantiene buena generalización y baja varianza.

    subsample=0.8,          # fracción de observaciones usadas en cada iteración
                            #Significa que cada árbol usa solo el 80 % de las filas de entrenamiento, seleccionadas aleatoriamente.
                            #Este “bagging parcial” introduce diversidad entre árboles y reduce la varianza.
                            #Si se pone en 1.0, usa todos los datos en cada iteración → más riesgo de sobreajuste.
                            #0.8 es un valor clásico que mejora la robustez sin perder demasiada información.
    colsample_bytree=0.8,   # fracción de columnas usadas en cada iteración
                            #Indica que cada árbol usa solo el 80 % de las columnas.
                            #También sirve para diversificar los árboles y evitar que todos dependan de las mismas variables 
                            # (por ejemplo, “Glucose”). Es equivalente a lo que hace Random Forest al usar subconjuntos de features.
                            #En datasets con menos de 10 variables (como este), 0.8 ayuda a evitar correlaciones 
                            # fuertes entre árboles.
    random_state=42,
    eval_metric='logloss'  #Define la métrica interna de evaluación durante el entrenamiento.
                            #En clasificación binaria, 'logloss' (logarithmic loss) es más sensible al desbalanceo de clases 
                            #que accuracy y penaliza más los errores de alta confianza.
                            #Es el estándar para calibrar la probabilidad de predicción (no solo la clase).
)

# Entrenar modelo
xgb.fit(X_train, y_train)

# Predicciones
y_pred_xgb = xgb.predict(X_test)

# Evaluación
print("Accuracy de XGBoost:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


#----------------------------------- OPTIMIZACION HIPER PARAMETROS

from sklearn.model_selection import GridSearchCV

param_grid_xgb = {
    'n_estimators': [100, 200, 300],   #Menos de 100 suele ser insuficiente para capturar relaciones complejas.
    'max_depth': [3, 4, 5, 6],         #Árboles muy profundos (>6) tienden a sobreajustar. Árboles muy superficiales (<3) pierden capacidad predictiva.
    'learning_rate': [0.01, 0.05, 0.1], #0.1 es el valor por defecto en XGBoost (aprendizaje rápido). 0.05 y 0.01 permiten un aprendizaje más lento pero más preciso y estable
    'subsample': [0.7, 0.8, 1.0],     #0.8 es un estándar muy usado.
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_xgb = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)

print("\nMejores hiperparámetros XGBoost:")
print(grid_xgb.best_params_)

best_xgb = grid_xgb.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)

print("\nEvaluación del modelo XGBoost optimizado:")
print("Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print(classification_report(y_test, y_pred_best_xgb))

#--------------------------------IMPORTANCIA DE LAS VARIABLES------------

importances_xgb = pd.Series(best_xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportancia de las VAriables XGBoost:")
print(importances_xgb)

importances_xgb.plot(kind="bar", figsize=(10,5), title="Importancia Variables (XGBoost)")
plt.show()

#------------------------------GUARAR MODELO

joblib.dump(best_xgb, "xgboost_optimized.pkl")
print("\nModelo XGBoost guardado como 'xgboost_optimized.pkl'")


# --------------------------------------------- COMPARAR MODELOS

# --- Métricas XGBoost ---
acc_xgb = accuracy_score(y_test, y_pred_best_xgb)
prec_xgb = precision_score(y_test, y_pred_best_xgb)
rec_xgb = recall_score(y_test, y_pred_best_xgb)
f1_xgb = f1_score(y_test, y_pred_best_xgb)

# --- Comparativa ---
print("\n================= COMPARACIÓN FINAL ENTRE MODELOS =================")
print(f"{'Métrica':<15}{'Decision Tree':<20}{'Random Forest':<20}{'XGBoost'}")
print(f"{'Accuracy':<15}{acc_tree:<20.4f}{acc_rf:<20.4f}{acc_xgb:.4f}")
print(f"{'Precision':<15}{prec_tree:<20.4f}{prec_rf:<20.4f}{prec_xgb:.4f}")
print(f"{'Recall':<15}{rec_tree:<20.4f}{rec_rf:<20.4f}{rec_xgb:.4f}")
print(f"{'F1-Score':<15}{f1_tree:<20.4f}{f1_rf:<20.4f}{f1_xgb:.4f}")

# --- Visualización ---
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
tree_scores = [acc_tree, prec_tree, rec_tree, f1_tree]
rf_scores = [acc_rf, prec_rf, rec_rf, f1_rf]
xgb_scores = [acc_xgb, prec_xgb, rec_xgb, f1_xgb]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9,5))
ax.bar(x - width, tree_scores, width, label='Decision Tree')
ax.bar(x, rf_scores, width, label='Random Forest')
ax.bar(x + width, xgb_scores, width, label='XGBoost')

ax.set_ylabel('Score')
ax.set_title('Comparación de modelos de clasificación (Diabetes)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()


#### OPTIMIZACION DE PARAMETROS Y CORRECCIONES 

""" scale_pos_weight → corrige desbalance de clases.
    early_stopping_rounds=20 → evita sobreajuste de árboles profundos.
    StratifiedKFold → mantiene proporción de clases en CV.
    scoring='f1' → prioriza balance entre recall y precision, ideal en medicina.
    Grid más amplio → explora mejor combinaciones de learning_rate, n_estimators, subsample y colsample_bytree."""

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ------------------------ Ajustar scale_pos_weight ------------------------
# Calcula el peso de la clase minoritaria
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ------------------------ Modelo base ------------------------
xgb_base = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
  )

# ------------------------ Grid de hiperparámetros ------------------------
param_grid_xgb = {
    'n_estimators': [200, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# ------------------------ Estrategia de validación cruzada ------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


grid_xgb = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    error_score='raise'
)

# ------------------------ Entrenamiento ------------------------
grid_xgb.fit(X_train, y_train)
print("\nMejores hiperparámetros XGBoost:")
print(grid_xgb.best_params_)



# ------------------------ Mejor modelo ------------------------
best_xgb = grid_xgb.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)

# ------------------------ Evaluación ------------------------
acc_xgb = accuracy_score(y_test, y_pred_best_xgb)
prec_xgb = precision_score(y_test, y_pred_best_xgb)
rec_xgb = recall_score(y_test, y_pred_best_xgb)
f1_xgb = f1_score(y_test, y_pred_best_xgb)

print("\n===== Evaluación XGBoost Optimizado =====")
print(f"Accuracy:  {acc_xgb:.4f}")
print(f"Precision: {prec_xgb:.4f}")
print(f"Recall:    {rec_xgb:.4f}")
print(f"F1-Score:  {f1_xgb:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best_xgb))

# ------------------------ Importancia de variables ------------------------
import pandas as pd
import matplotlib.pyplot as plt

importances_xgb = pd.Series(best_xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportancia de Variables (XGBoost Optimizado):")
print(importances_xgb)

importances_xgb.plot(kind="bar", figsize=(10,5), title="Importancia Variables (XGBoost Optimizado)")
plt.show()

# ------------------------ Guardar modelo ------------------------
import joblib
joblib.dump(best_xgb, "xgboost_optimized_v2.pkl")
print("\nModelo XGBoost optimizado guardado como 'xgboost_optimized_v2.pkl'")

#---------------------------------- COMPARACION FINAL--------------------

# ------------------------ Métricas ------------------------
models = {
    "Decision Tree": y_pred_best,
    "Random Forest": y_pred_best_rf,
    "XGBoost Base": y_pred_xgb,
    "XGBoost Optimizado": y_pred_best_xgb
}

metrics = {}
for name, y_pred in models.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics[name] = [acc, prec, rec, f1]

# ------------------------ Mostrar tabla ------------------------
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
print(f"{'Métrica':<15}{'Decision Tree':<18}{'Random Forest':<18}{'XGBoost Base':<18}{'XGBoost Optimizado'}")
for i, label in enumerate(labels):
    print(f"{label:<15}{metrics['Decision Tree'][i]:<18.4f}"
          f"{metrics['Random Forest'][i]:<18.4f}"
          f"{metrics['XGBoost Base'][i]:<18.4f}"
          f"{metrics['XGBoost Optimizado'][i]:.4f}")

# ------------------------ Visualización ------------------------
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x - 1.5*width, metrics['Decision Tree'], width, label='Decision Tree', color='skyblue')
ax.bar(x - 0.5*width, metrics['Random Forest'], width, label='Random Forest', color='orange')
ax.bar(x + 0.5*width, metrics['XGBoost Base'], width, label='XGBoost Base', color='green')
ax.bar(x + 1.5*width, metrics['XGBoost Optimizado'], width, label='XGBoost Optimizado', color='red')

ax.set_ylabel('Score')
ax.set_title('Comparación de Modelos de Clasificación (Diabetes)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Añadir valores encima de cada barra
for i in range(len(labels)):
    for j, key in enumerate(models.keys()):
        yval = metrics[key][i]
        ax.text(x[i] + (j-1.5)*width, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

#---- METRICAS MAS IMPORTANTES PARA OPTIMIZAR
#Métrica	Prioridad	Razón
#Recall	⭐⭐⭐⭐⭐	Evitar falsos negativos. No queremos decirle a alguien “estás bien” cuando está enfermo.
#F1-Score	⭐⭐⭐⭐	Balancea recall y precisión. Ideal cuando hay clases desbalanceadas (como diabetes).
#Es mejor detectar a alguien enfermo aunque sea una falsa alarma, que dejar pasar un caso real.

# OPTIMIZANDO RECALL Y F1

# Calcular peso de clases para favorecer Recall (clase positiva)
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
scale_pos_weight = neg / pos

xgb_recall = XGBClassifier(
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

param_dist_recall = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [2, 3, 4],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

from sklearn.model_selection import RandomizedSearchCV

random_search_recall = RandomizedSearchCV(
    estimator=xgb_recall,
    param_distributions=param_dist_recall,
    n_iter=25,                     # más bajo = más rápido, puedes subir a 50 si quieres máximo performance
    scoring='f1',                  # optimizamos F1
    cv=5,
    verbose=1,
    n_jobs=-1
)

random_search_recall.fit(
    X_train, y_train
)

best_xgb_recall = random_search_recall.best_estimator_

y_pred_recall = best_xgb_recall.predict(X_test)

print("\n===== XGBoost Optimizado para Recall & F1 =====")
print("Best Params:", random_search_recall.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_recall))
print("Recall:", recall_score(y_test, y_pred_recall))
print("F1 Score:", f1_score(y_test, y_pred_recall))
print("\nClassification Report:\n", classification_report(y_test, y_pred_recall))

# --- Métricas XGBoost ---
acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print("\n================= COMPARACIÓN FINAL ENTRE MODELOS =================")
print(f"{'Métrica':<15}{'Decision Tree':<18}{'Random Forest':<18}{'XGBoost'}")
print(f"{'Accuracy':<15}{acc_tree:<18.4f}{acc_rf:<18.4f}{acc_xgb:.4f}")
print(f"{'Precision':<15}{prec_tree:<18.4f}{prec_rf:<18.4f}{prec_xgb:.4f}")
print(f"{'Recall':<15}{rec_tree:<18.4f}{rec_rf:<18.4f}{rec_xgb:.4f}")
print(f"{'F1-Score':<15}{f1_tree:<18.4f}{f1_rf:<18.4f}{f1_xgb:.4f}")

# --- Visualización de comparación ---
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
tree_scores = [acc_tree, prec_tree, rec_tree, f1_tree]
rf_scores = [acc_rf, prec_rf, rec_rf, f1_rf]
xgb_scores = [acc_xgb, prec_xgb, rec_xgb, f1_xgb]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9,5))
bars1 = ax.bar(x - width, tree_scores, width, label='Decision Tree')
bars2 = ax.bar(x, rf_scores, width, label='Random Forest')
bars3 = ax.bar(x + width, xgb_scores, width, label='XGBoost')

ax.set_ylabel('Puntaje')
ax.set_title('Comparación de métricas entre modelos')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

#----------------------------------- OPTIMIZACION HIPER PARAMETROS

from sklearn.model_selection import GridSearchCV

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search_xgb = GridSearchCV(
    XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ),
    param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_xgb.fit(X_train, y_train)

print("\nMejores hiperparámetros XGBoost:")
print(grid_search_xgb.best_params_)

# Evaluar mejor modelo
best_xgb = grid_search_xgb.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)

print("\nEvaluación del XGBoost optimizado:")
print("Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print(classification_report(y_test, y_pred_best_xgb))

# Guardar modelo
joblib.dump(best_xgb, "xgboost_optimized.pkl")
print("\nModelo XGBoost guardado como 'xgboost_optimized.pkl'")

#----------------------------------- OPTIMIZACION HIPERPARÁMETROS XGBOOST

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

param_grid_xgb = {
    'n_estimators': [100, 200, 300],     # número de árboles
    'learning_rate': [0.01, 0.05, 0.1],  # velocidad de aprendizaje
    'max_depth': [3, 4, 5],              # profundidad del árbol
    'subsample': [0.7, 0.8, 1.0],        # fracción de filas
    'colsample_bytree': [0.7, 0.8, 1.0]  # fracción de columnas
}

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    cv=3,                 # 5 es más preciso pero más lento
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_xgb.fit(X_train, y_train)

print("\nMejores hiperparámetros de XGBoost:")
print(grid_search_xgb.best_params_)

best_xgb = grid_search_xgb.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)

print("\nEvaluación XGBoost Optimizado:")
print("Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print(classification_report(y_test, y_pred_best_xgb))

# Guardar modelo
joblib.dump(best_xgb, "xgboost_optimized.pkl")
print("\nModelo XGBoost guardado como 'xgboost_optimized.pkl'")

#------------------------- COMPARACIÓN FINAL ENTRE LOS 3 MODELOS

models = {
    "Decision Tree": (y_pred_best),
    "Random Forest": (y_pred_best_rf),
    "XGBoost": (y_pred_best_xgb)
}

print("\n=========== COMPARACIÓN FINAL MODELOS ===========")
for name, y_pred in models.items():
    print(f"\n📌 Modelo: {name}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")