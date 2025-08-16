import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("BDMM - Balanced Decision-Making Method")

# === Entrada de dados ===
num_decisores = st.number_input("Número de decisores:", min_value=1, value=2, step=1)
num_criterios = st.number_input("Número de critérios:", min_value=1, value=3, step=1)

# Nomes dos critérios
st.subheader("Defina os nomes dos critérios")
nomes_criterios = []
for i in range(num_criterios):
    criterio = st.text_input(f"Nome do critério {i+1}:", value=f"C{i+1}")
    nomes_criterios.append(criterio)

# Coleta dos pesos
st.subheader("Pesos atribuídos pelos decisores")
dados_pesos = {'Critério': nomes_criterios}
for i in range(num_decisores):
    decisor = f'D{i+1}'
    pesos = []
    st.markdown(f"**{decisor}**")
    for j in range(num_criterios):
        peso = st.number_input(
            f"Peso de {nomes_criterios[j]} ({decisor}):",
            min_value=0.0, max_value=1.0, value=0.1, step=0.01, key=f"{decisor}_{j}"
        )
        pesos.append(peso)
    dados_pesos[decisor] = pesos

df_pesos = pd.DataFrame(dados_pesos)

# Normalização automática se a soma não for 1
for i in range(num_decisores):
    decisor = f'D{i+1}'
    soma = df_pesos[decisor].sum()
    if not np.isclose(soma, 1.0):
        df_pesos[decisor] = df_pesos[decisor] / soma

st.write("### Pesos normalizados")
st.dataframe(df_pesos)

# === Passo 1: Pesos iguais ===
n = len(df_pesos['Critério'])
df_pesos['Pesos_Iguais'] = [1/n] * n

# === Passo 2: Distâncias Euclidianas ===
distancias_decisores = {}
distancia_total = 0
for i in range(num_decisores):
    decisor = f'D{i+1}'
    df_pesos[f'{decisor}_Diff'] = (df_pesos[decisor] - df_pesos['Pesos_Iguais'])**2
    distancia = np.sqrt(df_pesos[f'{decisor}_Diff'].sum())
    distancias_decisores[decisor] = distancia
    distancia_total += distancia

# Normalização das distâncias
pesos_normalizados = {f'Normalizado_{decisor}': dist/distancia_total for decisor, dist in distancias_decisores.items()}

# Resultados intermediários
resultados = {'Métrica': [], 'Valor': []}
for decisor, dist in distancias_decisores.items():
    resultados['Métrica'].append(f'{decisor}_Distância')
    resultados['Valor'].append(dist)
for nome_norm, peso_norm in pesos_normalizados.items():
    resultados['Métrica'].append(nome_norm)
    resultados['Valor'].append(peso_norm)

df_resultados = pd.DataFrame(resultados)
st.write("### Distâncias e Pesos Normalizados")
st.dataframe(df_resultados)

# === Passo 3: Pesos Combinados ===
df_pesos['Peso_Combinado'] = 0
for i in range(num_decisores):
    decisor = f'D{i+1}'
    nome_norm = f'Normalizado_{decisor}'
    df_pesos['Peso_Combinado'] += pesos_normalizados[nome_norm] * df_pesos[decisor]

st.write("### Pesos Combinados por Critério")
st.dataframe(df_pesos[['Critério', 'Peso_Combinado']])

# === Análise de Consenso via Desvio-Padrão ===
colunas_decisores = [f'D{i+1}' for i in range(num_decisores)]
desvio = df_pesos[colunas_decisores].std(axis=1, ddof=1)
media = df_pesos[colunas_decisores].mean(axis=1)
desvio_max = np.sqrt(media * (1 - media))
desvio_max_seguro = desvio_max.replace(0, np.nan)
disp_relativa = (desvio / desvio_max_seguro).clip(upper=1)
CI = 1 - disp_relativa

def classificar_ci(x):
    if x >= 0.85: return 'Alto consenso'
    if x >= 0.70: return 'Moderado'
    if x >= 0.50: return 'Baixo'
    return 'Dissenso'

df_consenso = pd.DataFrame({
    'Critério': df_pesos['Critério'],
    'Média dos Pesos': media,
    'Desvio-Padrão': desvio,
    'Índice de Consenso (CI)': CI
})
df_consenso['Nível de Consenso'] = df_consenso['Índice de Consenso (CI)'].apply(classificar_ci)

st.write("### Análise de Consenso por Critério")
st.dataframe(df_consenso)

# === Visualização Gráfica dos Pesos Combinados ===
st.write("### Gráfico de Pesos Combinados")
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(df_pesos['Critério'], df_pesos['Peso_Combinado'], color='steelblue')
ax.set_ylabel("Valor do Peso Combinado")
ax.set_title("Pesos Combinados Finais por Critério (BDMM)")
st.pyplot(fig)

# === Legenda das faixas de consenso ===
st.markdown("""
## O que é o consenso?

O **consenso** em processos de decisão multicritério é a **medida do quanto os decisores concordam entre si** ao atribuir pesos ou avaliar alternativas.  

- **Alto consenso:** todos os decisores têm opiniões muito próximas, indicando forte acordo.  
- **Moderado:** existe concordância, mas com pequenas divergências.  
- **Baixo ou Dissenso:** opiniões muito divergentes, indicando que os decisores não compartilham a mesma visão sobre a importância dos critérios.  

O desvio-padrão dos pesos atribuídos é usado para quantificar essa dispersão: quanto menor o desvio-padrão, maior o consenso.
""")

st.markdown("""
**Interpretação do índice de consenso (CI):**

- **CI ≥ 0,85:** Alto consenso  
- **0,70 ≤ CI < 0,85:** Moderado  
- **0,50 ≤ CI < 0,70:** Baixo  
- **CI < 0,50:** Dissenso
""")

