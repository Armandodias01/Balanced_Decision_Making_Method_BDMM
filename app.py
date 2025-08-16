import streamlit as st
import pandas as pd
import numpy as np

# Título do aplicativo
st.title("BDMM - Consolidação de Pesos e Medida de Consenso")

# Entrada do número de decisores e critérios
num_decisores = st.number_input("Número de decisores:", min_value=1, step=1, value=2)
num_criterios = st.number_input("Número de critérios:", min_value=1, step=1, value=3)

# Nomes dos critérios
nomes_criterios = []
st.subheader("Defina os nomes dos critérios")
for i in range(num_criterios):
    criterio = st.text_input(f"Nome do critério {i+1}:", f"C{i+1}")
    nomes_criterios.append(criterio)

# Pesos informados por cada decisor
dados_pesos = {'Critério': nomes_criterios}
for i in range(num_decisores):
    decisor = f'Decisor_{i+1}'
    pesos = []
    st.subheader(f"Pesos atribuídos pelo {decisor}")
    for j in range(num_criterios):
        peso = st.number_input(
            f"Peso de {nomes_criterios[j]} ({decisor}):",
            min_value=0.0,
            value=0.1,
            step=0.01,
            key=f"{decisor}_{j}"
        )
        pesos.append(peso)
    dados_pesos[decisor] = pesos

# Criação do DataFrame com os pesos
df_pesos = pd.DataFrame(dados_pesos)

st.write("### Tabela de Pesos Informados")
st.dataframe(df_pesos)

# Cálculo dos pesos iguais
n = len(df_pesos['Critério'])
pesos_iguais = [1 / n] * n
df_pesos['Pesos_Iguais'] = pesos_iguais

# Distâncias dos decisores em relação ao vetor de pesos iguais
distancias_decisores = {}
distancia_total = 0
for i in range(num_decisores):
    decisor = f'Decisor_{i+1}'
    df_pesos[f'{decisor}_Dif'] = (df_pesos[decisor] - df_pesos['Pesos_Iguais'])**2
    distancia = np.sqrt(df_pesos[f'{decisor}_Dif'].sum())
    distancias_decisores[decisor] = distancia
    distancia_total += distancia

# Normalização dos pesos dos decisores
pesos_normalizados = {f'Peso_{d}': dist/distancia_total for d, dist in distancias_decisores.items()}

# Medida de consenso: desvio-padrão dos pesos de cada critério
df_pesos['Desvio_Padrão'] = df_pesos.iloc[:, 1:1+num_decisores].std(axis=1)

# Resultados
st.write("### Distâncias e Pesos Normalizados dos Decisores")
st.json(pesos_normalizados)

st.write("### Tabela Final com Diferenças e Consenso")
st.dataframe(df_pesos)

st.write("#### Interpretação da Medida de Consenso")
st.markdown("""
- **Desvio-padrão baixo (~0 a 0,05):** Forte consenso entre os decisores.  
- **Desvio-padrão médio (0,05 a 0,15):** Consenso moderado.  
- **Desvio-padrão alto (> 0,15):** Forte divergência entre decisores.  
""")
