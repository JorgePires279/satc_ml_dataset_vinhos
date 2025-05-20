import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore') # Ignora warnings para um output mais limpo

# --- Configurações Iniciais ---
# Defina o caminho para o seu dataset
DATASET_PATH = 'data\group_4_winequality.csv' # Mude para o seu arquivo, ou combine se tiver brancos e tintos

# --- Carregamento dos Dados ---
print("--- Carregando os Dados ---")
try:
    df = pd.read_csv(DATASET_PATH)
    print("Dados carregados com sucesso!")
    print(f"Formato dos dados: {df.shape}")
    print("\nPrimeiras 5 linhas do dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"ERRO: Dataset não encontrado em '{DATASET_PATH}'. Verifique o caminho.")
    exit()

# --- 1. Análise Exploratória de Dados (AED) ---
print("\n--- 1. Análise Exploratória de Dados (AED) ---")
print("\nInformações gerais do dataset:")
df.info()

print("\nEstatísticas descritivas:")
print(df.describe())

# Histogramas para cada variável
print("\nGerando histogramas das variáveis...")
df.hist(bins=15, figsize=(15, 10))
plt.suptitle('Histogramas das Variáveis Físico-Químicas e Qualidade')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Scatter plots (exemplo para as 3 primeiras features vs quality)
print("\nGerando scatter plots (exemplo)...")
plt.figure(figsize=(15, 5))
for i, col in enumerate(df.columns[:-1][:3]): # Apenas as 3 primeiras features
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(x=col, y='quality', data=df, alpha=0.6)
    plt.title(f'{col} vs Quality')
plt.tight_layout()
plt.show()

# Matriz de Correlação
print("\nGerando matriz de correlação...")
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação')
plt.show()

# --- 2. Limpeza e Pré-processamento ---
print("\n--- 2. Limpeza e Pré-processamento ---")
print("\nVerificando valores ausentes:")
print(df.isnull().sum())
# Neste dataset, geralmente não há valores ausentes. Se houvesse, você trataria aqui.
# Ex: df.fillna(df.median(), inplace=True)

# Detecção de Outliers (usando IQR para 'quality' como exemplo)
print("\nDetecção de Outliers (Exemplo: Variável 'quality')")
Q1 = df['quality'].quantile(0.25)
Q3 = df['quality'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_quality = df[(df['quality'] < lower_bound) | (df['quality'] > upper_bound)]
print(f"Número de outliers em 'quality': {len(outliers_quality)}")
# Para este projeto, podemos optar por manter os outliers ou removê-los se forem poucos
# df_cleaned = df[~((df['quality'] < lower_bound) | (df['quality'] > upper_bound))]

# Separação de features (X) e target (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Padronização (StandardScaler é recomendado para modelos baseados em distância/gradiente)
print("\nPadronizando as features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Dados divididos: Treino ({X_train.shape[0]} amostras), Teste ({X_test.shape[0]} amostras)")

# --- 3. Modelagem (Regressão) ---
print("\n--- 3. Modelagem (Regressão) ---")
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    # 'Support Vector Regressor (SVR)': SVR() # SVR pode ser mais lento para datasets grandes
}

results_regression = {}

for name, model in models.items():
    print(f"\nTreinando {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- 4. Avaliação (Regressão) ---
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results_regression[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    print(f"Resultados para {name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

# --- 5. Ajuste de Hiperparâmetros (GridSearchCV exemplo para Random Forest) ---
print("\n--- 5. Ajuste de Hiperparâmetros (Regressão - GridSearchCV para Random Forest) ---")
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': [0.6, 0.8, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_reg = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_reg, param_grid=param_grid_rf,
                              cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

best_rf_reg = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_reg.predict(X_test)

best_rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
best_mae_rf = mean_absolute_error(y_test, y_pred_best_rf)
best_r2_rf = r2_score(y_test, y_pred_best_rf)

print("\nMelhores hiperparâmetros para Random Forest Regressor:")
print(grid_search_rf.best_params_)
print(f"Melhores resultados com Random Forest (tuned):")
print(f"  RMSE: {best_rmse_rf:.4f}")
print(f"  MAE: {best_mae_rf:.4f}")
print(f"  R²: {best_r2_rf:.4f}")

results_regression['Random Forest Regressor (Tuned)'] = {'RMSE': best_rmse_rf, 'MAE': best_mae_rf, 'R2': best_r2_rf}

# Comparativo de todos os resultados
print("\n--- Comparativo Final dos Modelos de Regressão ---")
for name, metrics in results_regression.items():
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# --- 6. Discussão (Regressão) ---
print("\n--- 6. Discussão (Regressão) ---")
# Interpretar coeficientes para modelos lineares
if 'Linear Regression' in models:
    lin_reg_model = models['Linear Regression']
    if hasattr(lin_reg_model, 'coef_'):
        print("\nCoeficientes da Regressão Linear:")
        coefficients = pd.Series(lin_reg_model.coef_, index=X.columns).sort_values(ascending=False)
        print(coefficients)
        print("\nVariáveis com maior impacto (Regressão Linear):")
        print(coefficients.head()) # As com maiores valores absolutos
        print(coefficients.tail()) # As com menores valores absolutos (negativos)

# Feature Importance para modelos baseados em árvores (Random Forest)
if 'Random Forest Regressor' in models and hasattr(best_rf_reg, 'feature_importances_'):
    print("\nImportância das Features (Random Forest Regressor):")
    feature_importances = pd.Series(best_rf_reg.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_importances)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
    plt.title('Importância das Features (Random Forest Regressor)')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

print("\nAnálise crítica dos resultados da regressão:")
print("- Compare os RMSE, MAE e R² dos modelos. Qual se saiu melhor?")
print("- Os modelos baseados em árvores (Random Forest, Gradient Boosting) geralmente têm melhor desempenho do que a Regressão Linear em dados não lineares.")
print("- A importância das features revela quais características físico-químicas são mais preditivas para a qualidade do vinho.")
print("- A Regressão Linear nos dá insights sobre a direção do impacto (positivo/negativo) dos recursos nos preços (ex: 'alcohol' geralmente tem um coeficiente positivo na qualidade).")
print("- O ajuste de hiperparâmetros (GridSearchCV) geralmente melhora o desempenho do modelo, mas exige mais tempo de processamento.")