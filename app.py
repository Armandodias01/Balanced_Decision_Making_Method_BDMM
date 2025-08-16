import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("BDMM - Consolidação de Pesos e Consenso")

# === Entrada de dados ===
num_decisores = st.number_input("Número de decisores:", min_value=1, value=2, step=1)
num_criterios = st.number_input("Número de critérios:", min_value=1, value=3, step=1)

# Nomes dos critérios
st.subheader("Defina os nomes dos critérios")
nomes_criterios = []
for i in range(num_criterios):
    criterio = st.text_input(f"Nome do critério {i+1}:", value=f"C{i+1}")
    nomes_criterios.append(criterio)

# Coleta dos pesos de cada decisor
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

# === Teste de soma dos pesos ===
for i in range(num_decisores):
    decisor = f'D{i+1}'
    soma = df_pesos[decisor].sum()
    if not np.isclose(soma, 1.0):
        st.warning(f"Atenção: os pesos de {decisor} não somam 1 (soma = {soma:.3f}). Normalizando...")
        df_pesos[decisor] = df_pesos[decisor] / soma

st.write("### Pesos informados / normalizados")
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

st.write("### Pesos Combinados")
st.dataframe(df_pesos[['Critério', 'Peso_Combinado']])

# === Análise de Consenso ===
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
    'Média_pesos': media,
    'Desvio_Padrão': desvio,
    'Desvio_Max(teórico)': desvio_max,
    'Dispersão_Relativa': disp_relativa.fillna(0),
    'Índice_Consenso_CI': CI.fillna(1)
})
df_consenso['Nível_Consenso'] = df_consenso['Índice_Consenso_CI'].apply(classificar_ci)

st.write("### Análise de Consenso")
st.dataframe(df_consenso)

# === Visualização gráfica ===
st.write("### Gráfico de Pesos Combinados")
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(df_pesos['Critério'], df_pesos['Peso_Combinado'], color='steelblue')
ax.set_ylabel("Valor do Peso Combinado")
ax.set_title("Pesos Combinados Finais por Critério (BDMM)")
st.pyplot(fig)
