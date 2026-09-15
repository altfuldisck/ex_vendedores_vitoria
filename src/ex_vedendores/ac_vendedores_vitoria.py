"""
AC vendedores 
nome do aluno: Vitória Altfuldisck Soares
matrícula: 20250016213
email: vitoria.altfuldisck@gmail.com
"""

# 1. Integração e qualidade dos dados
# 1. Leia os dois arquivos CSV em DataFrames do Pandas.
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
df_desempenho = pd.read_csv('vendedores_desempenho.csv')
df_contexto = pd.read_csv('vendedores_contexto.csv')
# 2. Informe a quantidade de linhas e colunas de cada base.
print("Desempenho:", df_desempenho.shape)
print("Contexto:", df_contexto.shape)
# 3. Identifique qual coluna deve ser utilizada como chave para integrar as duas bases.

# 4. Verifique se a chave identificada é única em cada base.
print("Único em Desempenho:", df_desempenho['id_vendedor'].is_unique)
print("Único em Contexto:", df_contexto['id_vendedor'].is_unique)
# 5. Faça o merge das duas bases preservando todos os vendedores da base de desempenho.
df_integrado = pd.merge(df_desempenho, df_contexto, on='id_vendedor', how='left')
# 6. Informe a quantidade de registros após o merge.
print("Registros no modelo integrado:", len(df_integrado))
# 7. Verifique a quantidade de valores ausentes em cada variável da base integrada.
print(df_integrado.isnull().sum())
# 8. Identifique quais variáveis possuem dados faltantes e comente possíveis razões para isso.


# 2. Estatística descritiva
# 9. Calcule a média de vendas_mes.
vendas_mes = df_integrado['vendas_mes'].mean()
# 10. Calcule a mediana de vendas_mes.
vendas_mes = df_integrado['vendas_mes'].median()
# 11. Informe o valor mínimo e o valor máximo de vendas_mes.
v_min = df_integrado['vendas_mes'].min()
v_max = df_integrado['vendas_mes'].max()
# 12. Identifique possíveis outliers em vendas_mes usando o critério do intervalo interquartil (IQR).
# 13. Compare a média de vendas entre vendedores que recebem bônus e vendedores que não recebem bônus.
# 14. Calcule a correlação entre as variáveis
# 15. Interprete o sinal das correlações encontradas.

# 3. Regressão linear múltipla
# 16. Crie uma base para regressão contendo apenas as variáveis: vendas_mes, horas_treinamento, faltas, numero_clientes, distancia_empresa_km e salario.
# 17. Remova apenas as observações com dados ausentes nas variáveis que serão utilizadas no modelo.
# 18. Informe quantas observações foram utilizadas na regressão.
# 19. Defina vendas_mes como variável dependente.
# 20. Utilize como variáveis explicativas: horas_treinamento, faltas, numero_clientes, distancia_empresa_km e salario.
# 21. Adicione a constante ao modelo e estime uma regressão linear múltipla utilizando Statsmodels.
# 22. Apresente o resumo completo da regressão.
# 23. Interprete o coeficiente de horas_treinamento, mantendo as demais variáveis constantes.
# 24. Interprete o coeficiente de faltas, mantendo as demais variáveis constantes.
# 25. Interprete o coeficiente de numero_clientes, mantendo as demais variáveis constantes.
# 26. Interprete o coeficiente de distancia_empresa_km.
# 27. Interprete o coeficiente de salario.
# 28. Apresente os p-valores de todas as variáveis explicativas.
# 29. Identifique quais variáveis são estatisticamente significativas ao nível de 5%.
# 30. Identifique quais variáveis não apresentam evidência estatística de associação com vendas_mes ao nível de 5%.
# 31. Informe o valor do R² do modelo.

# 4. Visualizações finais
# 32. Construa um histograma de vendas_mes utilizando Seaborn.
# 33. Construa um boxplot de vendas_mes utilizando Seaborn.
# 34. Construa um gráfico de dispersão com linha de regressão entre numero_clientes e vendas_mes.
# 35. Construa um gráfico com os coeficientes da regressão e seus intervalos de confiança de 95%.
