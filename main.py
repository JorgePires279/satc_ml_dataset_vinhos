import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore') # Ignora avisos para uma saída mais limpa

# --- Configurações Iniciais ---
# Defina o caminho para o seu dataset
CAMINHO_DATASET = 'data\\group_4_winequality.csv'

# --- Carregamento dos Dados ---
print("--- Carregando os Dados ---")
try:
    # Ao carregar, já tentamos usar o ponto como separador decimal globalmente
    # Se alguns números usarem vírgula como decimal, precisaremos de um tratamento mais específico
    df_vinho = pd.read_csv(CAMINHO_DATASET)
    print("Dados carregados com sucesso!")
    print(f"Formato dos dados: {df_vinho.shape}")
    print("\nPrimeiras 5 linhas do dataset (original):")
    print(df_vinho.head())
except FileNotFoundError:
    print(f"ERRO: Dataset não encontrado em '{CAMINHO_DATASET}'. Verifique o caminho.")
    print("Certifique-se de que o arquivo 'group_4_winequality.csv' está dentro da pasta 'data' no mesmo diretório do seu script.")
    exit()

# --- Pré-processamento Inicial (Correção de Tipos e Formato) ---
print("\n--- Pré-processamento Inicial: Correção de Tipos e Formato ---")

# Colunas que deveriam ser numéricas mas podem ter valores string problemáticos
colunas_numericas_potencialmente_problema = ['chlorides', 'density', 'alcohol'] # Adicione outras se notar problemas no info()

for coluna in colunas_numericas_potencialmente_problema:
    if coluna in df_vinho.columns:
        # Tenta converter para numérico. Erros são convertidos para NaN.
        # Antes de converter, tentamos limpar caracteres estranhos que poderiam ser separadores de milhares ou múltiplos pontos.
        # O erro '100.333.333.333.333' sugere pontos extra.
        # Vamos tentar remover todos os pontos, exceto o último, se houver mais de um.
        # Isso é uma heurística; pode precisar de ajuste dependendo do padrão real dos dados.
        df_vinho[coluna] = df_vinho[coluna].astype(str).apply(
            lambda x: x.replace('.', '', x.count('.') - 1) if x.count('.') > 1 else x
        )
        df_vinho[coluna] = pd.to_numeric(df_vinho[coluna], errors='coerce')

print("\nInformações gerais do dataset após correção de tipos:")
df_vinho.info()

print("\nVerificando valores ausentes após a conversão de tipos:")
print(df_vinho.isnull().sum())

# Tratamento de valores ausentes (resultantes da conversão 'coerce' ou já existentes)
# Optamos por preencher com a mediana para não introduzir ruído de outliers na média
print("\nTratando valores ausentes com a mediana da coluna...")
for coluna in df_vinho.columns:
    if df_vinho[coluna].isnull().any():
        if df_vinho[coluna].dtype != 'object': # Apenas para colunas numéricas
            mediana_coluna = df_vinho[coluna].median()
            df_vinho[coluna].fillna(mediana_coluna, inplace=True)
            print(f"  Preenchido NaN na coluna '{coluna}' com a mediana ({mediana_coluna:.2f}).")
        else: # Para colunas categóricas (como 'type' ou 'quality' se forem convertidas para NaN por engano)
            # Para 'quality', se houver NaN, vamos remover as linhas, pois é nosso alvo
            if coluna == 'quality':
                contagem_nan_quality = df_vinho['quality'].isnull().sum()
                if contagem_nan_quality > 0:
                    df_vinho.dropna(subset=['quality'], inplace=True)
                    print(f"  Removidas {contagem_nan_quality} linhas com NaN na coluna 'quality' (alvo).")
            else: # Para outras colunas 'object', preencher com a moda ou 'desconhecido'
                moda_coluna = df_vinho[coluna].mode()[0]
                df_vinho[coluna].fillna(moda_coluna, inplace=True)
                print(f"  Preenchido NaN na coluna '{coluna}' com a moda ('{moda_coluna}').")

print("\nVerificando valores ausentes após tratamento:")
print(df_vinho.isnull().sum())
print(f"Formato dos dados após tratamento de NaN: {df_vinho.shape}")

# Após o tratamento de NaN e a correção de tipos, vamos verificar novamente as estatísticas
print("\nEstatísticas descritivas após pré-processamento inicial:")
print(df_vinho.describe())

# Reavaliando outliers extremos em 'density' e 'chlorides'
# Pelas estatísticas descritivas, density e chlorides ainda apresentam valores muito discrepantes
# (max de 3793 para density e 975 para chlorides, com medias muito menores).
# Isso sugere um problema de escala ou erro de medição para esses valores.
# Vamos aplicar um clipping (limitação) para valores extremamente altos/baixos
# Baseado na descrição do dataset original de vinho, valores típicos são:
# chlorides: 0.012 - 0.61 (aprox)
# density: 0.99 - 1.03 (aprox)
# Definindo limites razoáveis (pode ser ajustado após mais EDA)
limite_superior_chlorides = df_vinho['chlorides'].quantile(0.99) # 99º percentil para não cortar todos os valores mais altos
limite_inferior_chlorides = df_vinho['chlorides'].quantile(0.01) # 1º percentil
df_vinho['chlorides'] = np.clip(df_vinho['chlorides'], limite_inferior_chlorides, limite_superior_chlorides)

limite_superior_density = df_vinho['density'].quantile(0.99)
limite_inferior_density = df_vinho['density'].quantile(0.01)
# Para density, se houver valores na casa dos 900+ ou 9000+, eles devem ser divididos por 1000.
# O erro `100.333.333.333.333` sugere que o valor real pode ser `0.100...` ou `100.333...`.
# Vamos tentar uma correção heurística para density, supondo que valores muito altos foram digitados com erro de escala.
df_vinho['density'] = df_vinho['density'].apply(lambda x: x / 1000 if x > 100 else x) # Se a densidade é > 100, divide por 1000
df_vinho['density'] = np.clip(df_vinho['density'], limite_inferior_density, limite_superior_density)

print("\nEstatísticas descritivas após tratamento de outliers extremos em chlorides e density:")
print(df_vinho[['chlorides', 'density']].describe())

# --- Começo da Parte 1 — Regressão (Predição de Nota de Qualidade) ---
# ... (O restante do código da Parte 1 e Parte 2 é basicamente o mesmo, mas agora
#      operando sobre o df_vinho limpo e com tipos corretos) ...

# --- 1. Análise Exploratória de Dados (AED) ---
# (Repetição para garantir que os gráficos usem os dados limpos)
print("\n--- 1. Análise Exploratória de Dados (AED - com dados limpos) ---")
# Histogramas para cada variável
print("\nGerando histogramas das variáveis (dados limpos)...")
df_vinho.hist(bins=15, figsize=(18, 12))
plt.suptitle('Histogramas das Variáveis Físico-Químicas e Qualidade do Vinho (Dados Limpos)', y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

# Matriz de Correlação (agora deve funcionar)
print("\nGerando matriz de correlação (dados limpos)...")
plt.figure(figsize=(12, 10))
# Precisamos garantir que todas as colunas sejam numéricas antes de calcular a correlação
# 'type' é categórica, então deve ser excluída da correlação numérica
colunas_numericas_para_corr = df_vinho.select_dtypes(include=np.number).columns
sns.heatmap(df_vinho[colunas_numericas_para_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação das Variáveis do Vinho (Dados Limpos)')
plt.show()


# --- 2. Limpeza e Pré-processamento (Continuação) ---
print("\n--- 2. Limpeza e Pré-processamento (Continuação) ---")

# A coluna 'type' é categórica e precisa ser convertida para numérica para os modelos
# Usaremos OneHotEncoder para 'type', pois não há ordem inerente (red vs white)
df_vinho_processado = pd.get_dummies(df_vinho, columns=['type'], drop_first=True, prefix='type')
print("\nDataset após One-Hot Encoding para a coluna 'type':")
print(df_vinho_processado.head())

# Separação de features (X) e target (y)
# 'quality' é o alvo. 'type_white' é a nova feature.
caracteristicas = df_vinho_processado.drop('quality', axis=1)
qualidade_alvo = df_vinho_processado['quality']

# Padronização (StandardScaler)
print("\nPadronizando as features...")
escalador = StandardScaler()
caracteristicas_escaladas = escalador.fit_transform(caracteristicas)
caracteristicas_escaladas = pd.DataFrame(caracteristicas_escaladas, columns=caracteristicas.columns)

# Divisão em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(caracteristicas_escaladas, qualidade_alvo, test_size=0.2, random_state=42)
print(f"Dados divididos: Treino ({X_treino.shape[0]} amostras), Teste ({X_teste.shape[0]} amostras)")

# --- 3. Modelagem (Regressão) ---
print("\n--- 3. Modelagem (Regressão) ---")
modelos_regressao = {
    'Regressão Linear': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
}

resultados_regressao = {}

for nome_modelo, modelo in modelos_regressao.items():
    print(f"\nTreinando {nome_modelo}...")
    modelo.fit(X_treino, y_treino)
    y_predito = modelo.predict(X_teste)

    # --- 4. Avaliação (Regressão) ---
    rmse = np.sqrt(mean_squared_error(y_teste, y_predito))
    mae = mean_absolute_error(y_teste, y_predito)
    r2 = r2_score(y_teste, y_predito)

    resultados_regressao[nome_modelo] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}

    print(f"Resultados para {nome_modelo}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

# --- 5. Ajuste de Hiperparâmetros (GridSearchCV exemplo para Random Forest Regressor) ---
print("\n--- 5. Ajuste de Hiperparâmetros (Regressão - GridSearchCV para Random Forest) ---")
parametros_grid_rf = {
    'n_estimators': [100, 200], # Reduzindo para 2 para agilizar o exemplo
    'max_features': [0.6, 0.8], # Reduzindo para 2 para agilizar o exemplo
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_regressor = RandomForestRegressor(random_state=42)
busca_em_grade_rf = GridSearchCV(estimator=rf_regressor, param_grid=parametros_grid_rf,
                              cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
print("Iniciando busca em grade para Random Forest Regressor...")
busca_em_grade_rf.fit(X_treino, y_treino)

melhor_rf_regressor = busca_em_grade_rf.best_estimator_
y_predito_melhor_rf = melhor_rf_regressor.predict(X_teste)

melhor_rmse_rf = np.sqrt(mean_squared_error(y_teste, y_predito_melhor_rf))
melhor_mae_rf = mean_absolute_error(y_teste, y_predito_melhor_rf)
melhor_r2_rf = r2_score(y_teste, y_predito_melhor_rf)

print("\nMelhores hiperparâmetros para Random Forest Regressor:")
print(busca_em_grade_rf.best_params_)
print(f"Melhores resultados com Random Forest (ajustado):")
print(f"  RMSE: {melhor_rmse_rf:.4f}")
print(f"  MAE: {melhor_mae_rf:.4f}")
print(f"  R²: {melhor_r2_rf:.4f}")

resultados_regressao['Random Forest Regressor (Ajustado)'] = {'RMSE': melhor_rmse_rf, 'MAE': melhor_mae_rf, 'R²': melhor_r2_rf}

# Comparativo de todos os resultados de regressão
print("\n--- Comparativo Final dos Modelos de Regressão ---")
for nome, metricas in resultados_regressao.items():
    print(f"\n{nome}:")
    for nome_metrica, valor in metricas.items():
        print(f"  {nome_metrica}: {valor:.4f}")

# --- 6. Discussão (Regressão) ---
print("\n--- 6. Discussão (Regressão) ---")
# Interpretar coeficientes para modelos lineares
if 'Regressão Linear' in modelos_regressao:
    modelo_linear = modelos_regressao['Regressão Linear']
    if hasattr(modelo_linear, 'coef_'):
        print("\nCoeficientes da Regressão Linear:")
        # Colunas usadas para treino são as de caracteristicas_escaladas
        coeficientes = pd.Series(modelo_linear.coef_, index=caracteristicas_escaladas.columns).sort_values(ascending=False)
        print(coeficientes)
        print("\nVariáveis com maior impacto (Regressão Linear - em valor absoluto):")
        print(coeficientes.abs().sort_values(ascending=False).head(5))

# Importância das Features para modelos baseados em árvores (Random Forest)
if 'Random Forest Regressor' in modelos_regressao and hasattr(melhor_rf_regressor, 'feature_importances_'):
    print("\nImportância das Features (Random Forest Regressor):")
    # Colunas usadas para treino são as de caracteristicas_escaladas
    importancias_features = pd.Series(melhor_rf_regressor.feature_importances_, index=caracteristicas_escaladas.columns).sort_values(ascending=False)
    print(importancias_features)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias_features.values, y=importancias_features.index, palette='viridis')
    plt.title('Importância das Features (Random Forest Regressor)')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

print("\nAnálise crítica dos resultados da regressão:")
print("- Compare os RMSE, MAE e R² dos modelos. Qual se saiu melhor e por quê?")
print("- Os modelos baseados em árvores (Random Forest, Gradient Boosting) geralmente têm melhor desempenho do que a Regressão Linear em dados não lineares ou com interações complexas.")
print("- A importância das features revela quais características físico-químicas são mais preditivas para a qualidade do vinho. Quais se destacaram?")
print("- A Regressão Linear nos dá insights sobre a direção do impacto (positivo/negativo) dos recursos na qualidade. Por exemplo, 'alcohol' geralmente tem um coeficiente positivo.")
print("- O ajuste de hiperparâmetros (GridSearchCV) geralmente melhora o desempenho do modelo, mas exige mais tempo de processamento. Houve melhora significativa?")

# --- Parte 2 — Classificação (Faixas de Qualidade) ---
print("\n\n--- Parte 2 — Classificação (Faixas de Qualidade) ---")

# --- Requisito: Criar nova variável alvo categórica ---
def classificar_qualidade(qualidade_numerica):
    if qualidade_numerica <= 4:
        return 'Baixa'
    elif qualidade_numerica <= 6:
        return 'Média'
    else:
        return 'Alta'

print("\nDiscretizando a variável 'quality' para criar categorias de qualidade...")
df_vinho['categoria_qualidade'] = df_vinho['quality'].apply(classificar_qualidade)
print("Contagem de vinhos por categoria de qualidade:")
print(df_vinho['categoria_qualidade'].value_counts())

# Verificar distribuição das classes (desbalanceamento)
plt.figure(figsize=(8, 6))
sns.countplot(x='categoria_qualidade', data=df_vinho, palette='viridis',
              order=df_vinho['categoria_qualidade'].value_counts().index)
plt.title('Distribuição das Classes de Qualidade do Vinho')
plt.xlabel('Categoria de Qualidade')
plt.ylabel('Número de Vinhos')
plt.show()

# Discussão sobre desbalanceamento:
print("\n--- Análise de Desbalanceamento de Classes ---")
contagem_classes = df_vinho['categoria_qualidade'].value_counts()
print(contagem_classes)
tamanho_min_classe = contagem_classes.min()
tamanho_max_classe = contagem_classes.max()
if tamanho_max_classe / tamanho_min_classe > 2:
    print(f"ATENÇÃO: As classes estão desbalanceadas. A maior classe ({tamanho_max_classe}) é {tamanho_max_classe/tamanho_min_classe:.2f} vezes maior que a menor classe ({tamanho_min_classe}).")
    print("Considerar técnicas de balanceamento como Oversampling (SMOTE) ou Undersampling durante o pré-processamento de classificação.")
else:
    print("As classes parecem razoavelmente balanceadas ou o desbalanceamento não é severo o suficiente para exigir rebalanceamento imediato.")


# --- 1. Preparação (Classificação) ---
# Usar o df_vinho_processado que já tem 'type' codificado
caracteristicas_clf = df_vinho_processado.drop(['quality', 'categoria_qualidade'], axis=1, errors='ignore') # 'categoria_qualidade' ainda não está em df_vinho_processado
alvo_clf = df_vinho['categoria_qualidade']

# Codificação de rótulos (LabelEncoder para o target)
codificador_rotulos = LabelEncoder()
alvo_clf_codificado = codificador_rotulos.fit_transform(alvo_clf)
print(f"\nClasses codificadas: {list(codificador_rotulos.classes_)} -> {list(range(len(codificador_rotulos.classes_)))}")

# Padronização das features (reaplicar, pois as características podem ter mudado após one-hot encoding)
escalador_clf = StandardScaler()
caracteristicas_clf_escaladas = escalador_clf.fit_transform(caracteristicas_clf)
caracteristicas_clf_escaladas = pd.DataFrame(caracteristicas_clf_escaladas, columns=caracteristicas_clf.columns)

# Divisão em conjuntos de treino e teste (estratificado para manter a proporção das classes)
X_treino_clf, X_teste_clf, y_treino_clf, y_teste_clf = train_test_split(
    caracteristicas_clf_escaladas, alvo_clf_codificado, test_size=0.2, random_state=42, stratify=alvo_clf_codificado
)
print(f"Dados para classificação divididos: Treino ({X_treino_clf.shape[0]} amostras), Teste ({X_teste_clf.shape[0]} amostras)")
print("Distribuição das classes no treino (após split estratificado):")
valores_unicos, contagens = np.unique(y_treino_clf, return_counts=True)
for i, val in enumerate(valores_unicos):
    print(f"  Classe {codificador_rotulos.inverse_transform([val])[0]}: {contagens[i]} ({contagens[i]/len(y_treino_clf)*100:.2f}%)")

# Balanceamento de classes (descomentar e instalar imbalanced-learn se necessário)
# from imblearn.over_sampling import SMOTE
# if tamanho_max_classe / tamanho_min_classe > 2:
#     smote = SMOTE(random_state=42)
#     X_treino_clf_rebalanceado, y_treino_clf_rebalanceado = smote.fit_resample(X_treino_clf, y_treino_clf)
#     print(f"Formato dos dados de treino após SMOTE: {X_treino_clf_rebalanceado.shape}")
#     X_treino_clf = X_treino_clf_rebalanceado
#     y_treino_clf = y_treino_clf_rebalanceado
#     print("Distribuição das classes no treino (após SMOTE):")
#     valores_unicos, contagens = np.unique(y_treino_clf, return_counts=True)
#     for i, val in enumerate(valores_unicos):
#         print(f"  Classe {codificador_rotulos.inverse_transform([val])[0]}: {contagens[i]} ({contagens[i]/len(y_treino_clf)*100:.2f}%)")


# --- 2. Modelagem (Classificação) ---
print("\n--- 2. Modelagem (Classificação) ---")
modelos_classificacao = {
    'Regressão Logística': LogisticRegression(random_state=42, max_iter=1000),
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
}

resultados_classificacao = {}

for nome_modelo, modelo in modelos_classificacao.items():
    print(f"\nTreinando {nome_modelo}...")
    modelo.fit(X_treino_clf, y_treino_clf)
    y_predito_clf = modelo.predict(X_teste_clf)

    # --- 3. Avaliação (Classificação) ---
    acuracia = accuracy_score(y_teste_clf, y_predito_clf)
    precisao = precision_score(y_teste_clf, y_predito_clf, average='weighted', zero_division=0)
    recall = recall_score(y_teste_clf, y_predito_clf, average='weighted', zero_division=0)
    f1 = f1_score(y_teste_clf, y_predito_clf, average='weighted', zero_division=0)
    matriz_confusao = confusion_matrix(y_teste_clf, y_predito_clf)
    relatorio_classificacao = classification_report(y_teste_clf, y_predito_clf, target_names=codificador_rotulos.classes_, zero_division=0)

    resultados_classificacao[nome_modelo] = {
        'Acurácia': acuracia,
        'Precisão (Ponderada)': precisao,
        'Recall (Ponderado)': recall,
        'F1-Score (Ponderado)': f1,
        'Matriz de Confusão': matriz_confusao,
        'Relatório de Classificação': relatorio_classificacao
    }

    print(f"Resultados para {nome_modelo}:")
    print(f"  Acurácia: {acuracia:.4f}")
    print(f"  Precisão (Ponderada): {precisao:.4f}")
    print(f"  Recall (Ponderado): {recall:.4f}")
    print(f"  F1-Score (Ponderado): {f1:.4f}")
    print("\nRelatório de Classificação:\n", relatorio_classificacao)
    print("Matriz de Confusão:\n", matriz_confusao)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues',
                xticklabels=codificador_rotulos.classes_, yticklabels=codificador_rotulos.classes_)
    plt.title(f'Matriz de Confusão para {nome_modelo}')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.show()

# --- 4. Ajuste de Hiperparâmetros (GridSearchCV para Random Forest Classifier) ---
print("\n--- 4. Ajuste de Hiperparâmetros (Classificação - GridSearchCV para Random Forest) ---")
parametros_grid_rf_clf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

rf_classificador = RandomForestClassifier(random_state=42)
busca_em_grade_rf_clf = GridSearchCV(estimator=rf_classificador, param_grid=parametros_grid_rf_clf,
                                  cv=3, n_jobs=-1, verbose=1, scoring='f1_weighted')
print("Iniciando busca em grade para Random Forest Classifier...")
busca_em_grade_rf_clf.fit(X_treino_clf, y_treino_clf)

melhor_rf_classificador = busca_em_grade_rf_clf.best_estimator_
y_predito_melhor_rf_clf = melhor_rf_classificador.predict(X_teste_clf)

melhor_acuracia_rf_clf = accuracy_score(y_teste_clf, y_predito_melhor_rf_clf)
melhor_f1_rf_clf = f1_score(y_teste_clf, y_predito_melhor_rf_clf, average='weighted', zero_division=0)
melhor_relatorio_clf_rf_clf = classification_report(y_teste_clf, y_predito_melhor_rf_clf, target_names=codificador_rotulos.classes_, zero_division=0)
melhor_matriz_confusao_rf_clf = confusion_matrix(y_teste_clf, y_predito_melhor_rf_clf)

resultados_classificacao['Random Forest Classifier (Ajustado)'] = {
    'Acurácia': melhor_acuracia_rf_clf,
    'F1-Score (Ponderado)': melhor_f1_rf_clf,
    'Relatório de Classificação': melhor_relatorio_clf_rf_clf,
    'Matriz de Confusão': melhor_matriz_confusao_rf_clf
}

print("\nMelhores hiperparâmetros para Random Forest Classifier:")
print(busca_em_grade_rf_clf.best_params_)
print(f"Melhores resultados com Random Forest Classifier (ajustado):")
print(f"  Acurácia: {melhor_acuracia_rf_clf:.4f}")
print(f"  F1-Score (Ponderado): {melhor_f1_rf_clf:.4f}")
print("\nRelatório de Classificação (Ajustado):\n", melhor_relatorio_clf_rf_clf)
print("Matriz de Confusão (Ajustado):\n", melhor_matriz_confusao_rf_clf)

plt.figure(figsize=(8, 6))
sns.heatmap(melhor_matriz_confusao_rf_clf, annot=True, fmt='d', cmap='Blues',
            xticklabels=codificador_rotulos.classes_, yticklabels=codificador_rotulos.classes_)
plt.title('Matriz de Confusão para Random Forest Classifier (Ajustado)')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()

# --- 5. Discussão (Classificação) ---
print("\n--- 5. Discussão (Classificação) ---")
print("\nAnálise dos erros (Matrizes de Confusão):")
print("- Observe as matrizes de confusão. As células fora da diagonal principal indicam erros de classificação.")
print("- Quais classes foram mais frequentemente confundidas entre si? Por exemplo, vinhos 'Média' foram classificados como 'Baixa' ou 'Alta'? Isso pode indicar que as características físico-químicas não são distintivas o suficiente para diferenciar essas classes ou que o modelo precisa de mais dados/ajustes.")

print("\nComparativo de Modelos de Classificação:")
print("- Qual modelo se saiu melhor em termos de acurácia, F1-score, precisão e recall? Um F1-score alto é desejável, especialmente com classes desbalanceadas.")

if 'Random Forest Classifier' in modelos_classificacao and hasattr(melhor_rf_classificador, 'feature_importances_'):
    print("\nImportância das Features (Random Forest Classifier):")
    importancias_features_clf = pd.Series(melhor_rf_classificador.feature_importances_, index=caracteristicas_clf.columns).sort_values(ascending=False)
    print(importancias_features_clf)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias_features_clf.values, y=importancias_features_clf.index, palette='viridis')
    plt.title('Importância das Features (Random Forest Classifier)')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    print("\nAlguma feature foi decisiva para a classificação? Compare com a análise de regressão.")

print("\nConsiderações Finais do Projeto:")
print("- Reflita sobre as escolhas de faixas para a classificação. Como diferentes faixas (ranges) impactam o balanceamento e o desempenho do modelo?")
print("- Quais são as limitações dos modelos que você usou? Eles seriam adequados para uso em produção?")
print("- Sugestões para trabalhos futuros: Mais dados, engenharia de features mais elaborada (criação de novas features a partir das existentes), explorar outros modelos (e.g., redes neurais), validação cruzada mais robusta.")