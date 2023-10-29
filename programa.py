# Pré-requisitos para compilar e executar : 
# * Pyhton 3.11: obtido pela Microsoft Store
# * modules: 
#   * pandas: pip install pandas
#   * matplotlib: pip install matplotlib
#   * scikit-learn: pip install scikit-learn

# Importar as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

df = pd.read_csv("job.csv", encoding="ISO-8859-1")

# Converter a coluna 'PIB, (US$ atual)' para um formato numérico
df['PIB, (US$ atual)'] = df['PIB, (US$ atual)'].str.replace('[\$,\.]', '', regex=True).astype(float)

# Converter a coluna 'Expectativa de vida no nascimento, total (anos)' removendo vírgulas e convertendo para float
df['Expectativa de vida no nascimento, total (anos)'] = df['Expectativa de vida no nascimento, total (anos)'].str.replace(',', '').astype(float)

# Calcular o crescimento do PIB
df['Crescimento PIB'] = df['PIB, (US$ atual)'].pct_change() * 100

# Filtrar os dados para os últimos 10 anos
df_filtered = df[df['Anos'] >= (max(df['Anos']) - 10)]

# Informações adicionais hardcoded
correlacao_exp_vida_populacao = 0.995  
teste_normalidade_pib = 0.201  
teste_normalidade_exp_vida = 0.079  

# Ajustar a expectativa de vida com base na correlação com a população, e aplicar transformações baseadas nos testes de normalidade
df_filtered['Expectativa de vida ajustada'] = df_filtered['Expectativa de vida no nascimento, total (anos)'] * correlacao_exp_vida_populacao
if teste_normalidade_pib < 0.05:
    df_filtered['Crescimento PIB'] = np.log(df_filtered['Crescimento PIB'] + 1)
if teste_normalidade_exp_vida < 0.05:
    df_filtered['Expectativa de vida ajustada'] = np.log(df_filtered['Expectativa de vida ajustada'] + 1)

# Regressão Linear utilizando 'Crescimento PIB' e 'Expectativa de vida ajustada':
X = df_filtered['Crescimento PIB'].values.reshape(-1, 1)
y = df_filtered['Expectativa de vida ajustada'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Gerar um gráfico de dispersão e a linha de regressão
plt.scatter(X, y, color='blue')
for i, txt in enumerate(df_filtered['Anos']):
    plt.annotate(txt, (X[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.title('Regressão Linear entre Crescimento do PIB e Expectativa de Vida Ajustada nos Últimos 10 Anos')
plt.xlabel('Crescimento do PIB (%)')
plt.ylabel('Expectativa de Vida Ajustada (anos)')
plt.show()

# Calcular o coeficiente de determinação (R²)
r_squared = r2_score(y, y_pred)
print(f"Coeficiente de determinação (R^2): {r_squared:.2f}")














