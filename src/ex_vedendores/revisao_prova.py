#exercício 1

import pandas as pd
# Leitura dos arquivos CSV
df_desempenho = pd.read_csv("vendedores_desempenho.csv")
df_contexto = pd.read_csv("vendedores_contexto.csv")
# 1. Quantidade de linhas e colunas
print(f"Desempenho: {df_desempenho.shape[0]} linhas x {df_desempenho.shape[1]} colunas")
print(f"Contexto: {df_contexto.shape[0]} linhas x {df_contexto.shape[1]} colunas")
# 2. Chave de integração
chave = "id_vendedor"
# 3. Verificação de unicidade da chave
unicidade_desempenho = df_desempenho[chave].is_unique
unicidade_contexto = df_contexto[chave].is_unique
print(f"Chave única na base de desempenho? {unicidade_desempenho}")
print(f"Chave única na base de contexto? {unicidade_contexto}")
# 4. Merge preservando todos os vendedores de desempenho (Left Join)
df_integrado = pd.merge(df_desempenho, df_contexto, on=chave, how="left")
# 5. Quantidade de registros após o merge
print(f"Quantidade de registros após o merge: {len(df_integrado)}")
# 6. Verificação de valores ausentes em cada variável
faltantes = df_integrado.isnull().sum()
print("\nValores ausentes por variável:")
print(faltantes)

#exercicio 2
import numpy as np
# 1. Média, Mediana, Mínimo e Máximo de vendas_mes
media_vendas = df_integrado["vendas_mes"].mean()
mediana_vendas = df_integrado["vendas_mes"].median()
min_vendas = df_integrado["vendas_mes"].min()
max_vendas = df_integrado["vendas_mes"].max()
print(f"Média de vendas: {media_vendas:.2f}")
print(f"Mediana de vendas: {mediana_vendas:.2f}")
print(f"Mínimo: {min_vendas:.2f} | Máximo: {max_vendas:.2f}")
# 2. Identificação de outliers pelo critério do IQR
q1 = df_integrado["vendas_mes"].quantile(0.25)
q3 = df_integrado["vendas_mes"].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr
outliers = df_integrado[
    (df_integrado["vendas_mes"] < limite_inferior) | 
    (df_integrado["vendas_mes"] > limite_superior)
]
print(f"Limites IQR: [{limite_inferior:.2f}, {limite_superior:.2f}]")
print(f"Número de outliers em vendas_mes: {len(outliers)}")
# 3. Comparação da média de vendas por recebimento de bônus
media_por_bonus = df_integrado.groupby("bonus")["vendas_mes"].mean()
print("\nMédia de vendas por recebimento de bônus:")
print(media_por_bonus)
# 4. Correlação entre variáveis numéricas
vars_num = df_integrado.select_dtypes(include=[np.number]).columns
matriz_correlacao = df_integrado[vars_num].corr()
print("\nMatriz de Correlação:")
print(matriz_correlacao)

#exercicio 3
import statsmodels.api as sm
# 1. Criação da base para regressão
vars_modelo = [
    "vendas_mes",
    "horas_treinamento",
    "faltas",
    "numero_clientes",
    "distancia_empresa_km",
    "salario"
]
# 2. Remoção de dados ausentes nas variáveis do modelo
df_regressao = df_integrado[vars_modelo].dropna()
# 3. Quantidade de observações utilizadas
n_obs = len(df_regressao)
print(f"Observações utilizadas no modelo: {n_obs}")
# 4. Definição da variável dependente (Y) e explicativas (X)
Y = df_regressao["vendas_mes"]
X = df_regressao[[
    "horas_treinamento",
    "faltas",
    "numero_clientes",
    "distancia_empresa_km",
    "salario"
]]
# 5. Adição da constante e estimação do modelo OLS
X = sm.add_constant(X)
modelo = sm.OLS(Y, X).fit()
# 6. Resumo completo do modelo
print(modelo.summary())
# Exibição de p-valores e R²
print("\nP-valores das variáveis explicativas:")
print(modelo.pvalues)
print(f"\nR² do modelo: {modelo.rsquared:.4f}")

#exercicio 4
import matplotlib.pyplot as plt
import seaborn as sns
# Configuração estética geral
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# 1. Histograma de vendas_mes
sns.histplot(
    data=df_integrado, 
    x="vendas_mes", 
    kde=True, 
    ax=axes[0, 0], 
    color="skyblue"
)
axes[0, 0].set_title("Histograma de Vendas no Mês")
axes[0, 0].set_xlabel("Vendas Mês")
axes[0, 0].set_ylabel("Frequência")
# 2. Boxplot de vendas_mes
sns.boxplot(
    data=df_integrado, 
    y="vendas_mes", 
    ax=axes[0, 1], 
    color="lightgreen"
)
axes[0, 1].set_title("Boxplot de Vendas no Mês")
axes[0, 1].set_ylabel("Vendas Mês")
# 3. Gráfico de dispersão com linha de regressão (numero_clientes vs vendas_mes)
sns.regplot(
    data=df_integrado, 
    x="numero_clientes", 
    y="vendas_mes", 
    ax=axes[1, 0], 
    scatter_kws={"alpha": 0.6}, 
    line_kws={"color": "red"}
)
axes[1, 0].set_title("Número de Clientes vs. Vendas no Mês")
axes[1, 0].set_xlabel("Número de Clientes")
axes[1, 0].set_ylabel("Vendas Mês")
# 4. Gráfico dos coeficientes da regressão e intervalos de confiança (95%)
coef_df = modelo.params.drop("const").reset_index()
coef_df.columns = ["Variavel", "Coeficiente"]
conf_int = modelo.conf_int().loc[coef_df["Variavel"]]
coef_df["erro_inf"] = coef_df["Coeficiente"] - conf_int[0]
coef_df["erro_sup"] = conf_int[1] - coef_df["Coeficiente"]
axes[1, 1].errorbar(
    x=coef_df["Coeficiente"],
    y=coef_df["Variavel"],
    xerr=[coef_df["erro_inf"], coef_df["erro_sup"]],
    fmt="o",
    color="darkblue",
    ecolor="red",
    capsize=5
)
axes[1, 1].axvline(x=0, color="gray", linestyle="--")
axes[1, 1].set_title("Coeficientes da Regressão (IC 95%)")
axes[1, 1].set_xlabel("Valor do Coeficiente")
plt.tight_layout()
plt.show()