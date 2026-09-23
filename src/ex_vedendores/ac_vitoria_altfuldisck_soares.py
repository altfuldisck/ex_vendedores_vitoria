#Vitória Altfuldisck Soares

# ============================================================
# 1. INTEGRAÇÃO E QUALIDADE DOS DADOS
# ============================================================

# 1. Leia os dois arquivos CSV em DataFrames do Pandas.

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

df_desempenho = pd.read_csv("vendedores_desempenho.csv")
df_contexto = pd.read_csv("vendedores_contexto.csv")


# 2. Informe a quantidade de linhas e colunas de cada base.

print("Desempenho:", df_desempenho.shape)
print("Contexto:", df_contexto.shape)


# 3. Identifique qual coluna deve ser utilizada como chave para integrar as duas bases.

print("Colunas da base de desempenho:")
print(df_desempenho.columns.tolist())

print("\nColunas da base de contexto:")
print(df_contexto.columns.tolist())

chave = "id_vendedor"

print("\nChave utilizada:", chave)


# 4. Verifique se a chave identificada é única em cada base.

print(
    "Único em Desempenho:",
    df_desempenho["id_vendedor"].is_unique
)

print(
    "Único em Contexto:",
    df_contexto["id_vendedor"].is_unique
)


# 5. Faça o merge das duas bases preservando todos os vendedores
#    da base de desempenho.

df_integrado = pd.merge(
    df_desempenho,
    df_contexto,
    on="id_vendedor",
    how="left"
)

print("\nBase integrada:")
print(df_integrado)


# 6. Informe a quantidade de registros após o merge.

print(
    "\nRegistros no modelo integrado:",
    len(df_integrado)
)


# 7. Verifique a quantidade de valores ausentes em cada variável
#    da base integrada.

print("\nValores ausentes por variável:")
print(df_integrado.isnull().sum())


# 8. Identifique quais variáveis possuem dados faltantes
#    e comente possíveis razões para isso.

faltantes = df_integrado.isnull().sum()

print("\nVariáveis com dados faltantes:")
print(faltantes[faltantes > 0])

print(
    "\nPossíveis razões para os dados faltantes: "
    "informações não cadastradas, ausência de registro na base "
    "de contexto ou problemas de integração entre as bases."
)


# ============================================================
# 2. ESTATÍSTICA DESCRITIVA
# ============================================================

# 9. Calcule a média de vendas_mes.

media_vendas = df_integrado["vendas_mes"].mean()

print("\nMédia de vendas_mes:", media_vendas)


# 10. Calcule a mediana de vendas_mes.

mediana_vendas = df_integrado["vendas_mes"].median()

print("Mediana de vendas_mes:", mediana_vendas)


# 11. Informe o valor mínimo e o valor máximo de vendas_mes.

v_min = df_integrado["vendas_mes"].min()
v_max = df_integrado["vendas_mes"].max()

print("Valor mínimo de vendas_mes:", v_min)
print("Valor máximo de vendas_mes:", v_max)


# 12. Identifique possíveis outliers em vendas_mes
#     usando o critério do intervalo interquartil (IQR).

Q1 = df_integrado["vendas_mes"].quantile(0.25)
Q3 = df_integrado["vendas_mes"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Limite inferior:", limite_inferior)
print("Limite superior:", limite_superior)

outliers = df_integrado[
    (df_integrado["vendas_mes"] < limite_inferior) |
    (df_integrado["vendas_mes"] > limite_superior)
]

print("\nOutliers:")
print(outliers)


# 13. Compare a média de vendas entre vendedores que recebem
#     bônus e vendedores que não recebem bônus.

media_bonus = df_integrado.groupby("bonus")["vendas_mes"].mean()

print("\nMédia de vendas por bônus:")
print(media_bonus)


# 14. Calcule a correlação entre as variáveis.

correlacao = df_integrado.select_dtypes(
    include="number"
).corr()

print("\nMatriz de correlação:")
print(correlacao)


# 15. Interprete o sinal das correlações encontradas.

print("\nInterpretação das correlações:")

print(
    "Correlação positiva: quando uma variável aumenta, "
    "a outra tende a aumentar também."
)

print(
    "Correlação negativa: quando uma variável aumenta, "
    "a outra tende a diminuir."
)

print(
    "Correlação próxima de zero: existe pouca associação "
    "linear entre as variáveis."
)

print(
    "Importante: correlação não significa necessariamente causalidade."
)


# ============================================================
# 3. REGRESSÃO LINEAR MÚLTIPLA
# ============================================================

# 16. Crie uma base para regressão contendo apenas as variáveis:
#     vendas_mes, horas_treinamento, faltas, numero_clientes,
#     distancia_empresa_km e salario.

dados = df_integrado[
    [
        "vendas_mes",
        "horas_treinamento",
        "faltas",
        "numero_clientes",
        "distancia_empresa_km",
        "salario"
    ]
]

print("\nDados utilizados na regressão:")
print(dados)


# 17. Remova apenas as observações com dados ausentes
#     nas variáveis que serão utilizadas no modelo.

dados = dados.dropna()

print("\nDados após remoção dos valores ausentes:")
print(dados)


# 18. Informe quantas observações foram utilizadas na regressão.

print(
    "\nQuantidade de observações utilizadas na regressão:",
    len(dados)
)


# 19. Defina vendas_mes como variável dependente.

y = dados["vendas_mes"]

print("\nVariável dependente:")
print(y)


# 20. Utilize como variáveis explicativas:
#     horas_treinamento, faltas, numero_clientes,
#     distancia_empresa_km e salario.

X = dados[
    [
        "horas_treinamento",
        "faltas",
        "numero_clientes",
        "distancia_empresa_km",
        "salario"
    ]
]

print("\nVariáveis explicativas:")
print(X)


# 21. Adicione a constante ao modelo e estime uma regressão
#     linear múltipla utilizando Statsmodels.

X = sm.add_constant(X)

modelo = sm.OLS(y, X).fit()

print("\nRegressão estimada com sucesso!")


# 22. Apresente o resumo completo da regressão.

print("\nResumo da regressão:")
print(modelo.summary())


# 23. Interprete o coeficiente de horas_treinamento,
#     mantendo as demais variáveis constantes.

coef = modelo.params["horas_treinamento"]

print("\nCoeficiente de horas_treinamento:")
print(coef)

print(
    "Mantendo as demais variáveis constantes, "
    "um aumento de 1 hora de treinamento está associado "
    "a uma variação de",
    coef,
    "unidades em vendas_mes."
)


# 24. Interprete o coeficiente de faltas,
#     mantendo as demais variáveis constantes.

coef = modelo.params["faltas"]

print("\nCoeficiente de faltas:")
print(coef)

print(
    "Mantendo as demais variáveis constantes, "
    "um aumento de 1 falta está associado "
    "a uma variação de",
    coef,
    "unidades em vendas_mes."
)


# 25. Interprete o coeficiente de numero_clientes,
#     mantendo as demais variáveis constantes.

coef = modelo.params["numero_clientes"]

print("\nCoeficiente de numero_clientes:")
print(coef)

print(
    "Mantendo as demais variáveis constantes, "
    "um aumento de 1 cliente está associado "
    "a uma variação de",
    coef,
    "unidades em vendas_mes."
)


# 26. Interprete o coeficiente de distancia_empresa_km.

coef = modelo.params["distancia_empresa_km"]

print("\nCoeficiente de distancia_empresa_km:")
print(coef)

print(
    "Mantendo as demais variáveis constantes, "
    "um aumento de 1 km na distância está associado "
    "a uma variação de",
    coef,
    "unidades em vendas_mes."
)


# 27. Interprete o coeficiente de salario.

coef = modelo.params["salario"]

print("\nCoeficiente de salario:")
print(coef)

print(
    "Mantendo as demais variáveis constantes, "
    "um aumento de uma unidade no salário está associado "
    "a uma variação de",
    coef,
    "unidades em vendas_mes."
)


# 28. Apresente os p-valores de todas as variáveis explicativas.

pvalores = modelo.pvalues

print("\nP-valores:")
print(pvalores)


# 29. Identifique quais variáveis são estatisticamente
#     significativas ao nível de 5%.

print("\nVariáveis estatisticamente significativas a 5%:")

for variavel in X.columns:

    if variavel == "const":
        continue

    p = modelo.pvalues[variavel]

    if p < 0.05:
        print(
            variavel,
            "é estatisticamente significativa."
        )


# 30. Identifique quais variáveis não apresentam evidência
#     estatística de associação com vendas_mes ao nível de 5%.

print(
    "\nVariáveis sem evidência estatística de associação a 5%:"
)

for variavel in X.columns:

    if variavel == "const":
        continue

    p = modelo.pvalues[variavel]

    if p >= 0.05:
        print(
            variavel,
            "não apresenta evidência estatística "
            "de associação com vendas_mes."
        )


# 31. Informe o valor do R² do modelo.

R2 = modelo.rsquared

print("\nR² =", R2)

print(
    "O modelo explica aproximadamente",
    R2 * 100,
    "% da variação de vendas_mes."
)


# ============================================================
# 4. VISUALIZAÇÕES FINAIS
# ============================================================

# 32. Construa um histograma de vendas_mes utilizando Seaborn.

sns.histplot(
    data=df_integrado,
    x="vendas_mes",
    kde=True
)

plt.title("Histograma de vendas_mes")
plt.xlabel("Vendas por mês")
plt.ylabel("Frequência")

plt.show()


# 33. Construa um boxplot de vendas_mes utilizando Seaborn.

sns.boxplot(
    data=df_integrado,
    x="vendas_mes"
)

plt.title("Boxplot de vendas_mes")
plt.xlabel("Vendas por mês")

plt.show()


# 34. Construa um gráfico de dispersão com linha de regressão
#     entre numero_clientes e vendas_mes.

sns.regplot(
    data=df_integrado,
    x="numero_clientes",
    y="vendas_mes"
)

plt.title(
    "Número de clientes x vendas_mes"
)

plt.xlabel("Número de clientes")
plt.ylabel("Vendas por mês")

plt.show()


# 35. Construa um gráfico com os coeficientes da regressão
#     e seus intervalos de confiança de 95%.

coeficientes = modelo.params.drop("const")

intervalos = modelo.conf_int().loc[
    coeficientes.index
]

erro_inferior = (
    coeficientes - intervalos[0]
)

erro_superior = (
    intervalos[1] - coeficientes
)

plt.figure(figsize=(8, 5))

plt.errorbar(
    coeficientes.values,
    coeficientes.index,
    xerr=[
        erro_inferior.values,
        erro_superior.values
    ],
    fmt="o"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.title(
    "Coeficientes da regressão e IC de 95%"
)

plt.xlabel("Coeficiente")
plt.ylabel("Variável")

plt.show()