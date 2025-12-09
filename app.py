import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("BDMM - Balanced Decision-Making Method com Índice de Consenso")

st.markdown("""
Este aplicativo implementa o **BDMM (Balanced Decision-Making Method)**, combinando os pesos de múltiplos decisores(as) e calculando o **Índice de Consenso (IC)** para medir a concordância entre eles.
""")

# === Etapa 1: Entrada de dados ===
st.header("1. Entrada de Dados")
st.markdown("""
Cada decisor fornece um vetor de pesos para os critérios.  
Os pesos podem estar entre 0 e 1, e o BDMM fará a normalização automaticamente.
""")

num_decisores = st.number_input("Número de decisores:", min_value=1, value=2)
num_criterios = st.number_input("Número de critérios:", min_value=1, value=3)

nomes_criterios = []
for i in range(num_criterios):
    crit = st.text_input(f"Nome do critério {i+1}", value=f"C{i+1}")
    nomes_criterios.append(crit)

dados_pesos = {'Critério': nomes_criterios}

# Entrada dos pesos
for i in range(num_decisores):
    decisor = f'D{i+1}'
    pesos = []
    st.markdown(f"**Decisor {decisor}**")
    for j in range(num_criterios):
        p = st.number_input(f"Peso {nomes_criterios[j]} ({decisor})",
                            min_value=0.0, value=1/num_criterios,
                            step=0.01, key=f"{decisor}_{j}")
        pesos.append(p)
    dados_pesos[decisor] = pesos

df_pesos = pd.DataFrame(dados_pesos)

# === Normalização dos pesos (necessária para BDMM) ===
for i in range(num_decisores):
    decisor = f'D{i+1}'
    soma = df_pesos[decisor].sum()
    df_pesos[decisor] = df_pesos[decisor] / soma   # NORMALIZAÇÃO CORRETA

st.write("### Matriz de pesos")
st.dataframe(df_pesos)

# === Etapa 2: Vetor de Pesos Iguais ===
st.header("2. Vetor de Pesos Iguais")
m = len(df_pesos['Critério'])
df_pesos['Pesos_Iguais'] = 1/m

st.markdown(r"""
O vetor de pesos iguais é:

$$
w_{eq,l} = \frac{1}{m}
$$
""")

# === Etapa 3: Distância Euclidiana ===
st.header("3. Distância Euclidiana por decisor")
st.markdown(r"""
A distância Euclidiana mede o afastamento entre o vetor do decisor e o vetor igual:

$$
d_k = \sqrt{\sum_{l=1}^{m} (w'_{kl} - w_{eq})^2}
$$
""")

distancias = {}
for i in range(num_decisores):
    decisor = f'D{i+1}'
    dist = np.sqrt(((df_pesos[decisor] - df_pesos['Pesos_Iguais'])**2).sum())
    distancias[decisor] = dist

# === Etapa 4: Normalização das distâncias e pesos ajustados ===
st.header("4. Normalização das distâncias e cálculo dos pesos ajustados")

st.markdown(r"""
As distâncias são normalizadas para produzir pesos ajustados:

$$
ND_k = \frac{d_k}{\sum d_k}
$$

O peso ajustado:

$$
a_k = 1 - ND_k
$$
""")

dist_total = sum(distancias.values())

# EVITA DIVISÃO POR ZERO SE TODOS TIVEREM DISTÂNCIA ZERO
if dist_total == 0:
    pesos_ajustados = {dec: 1/num_decisores for dec in distancias}
else:
    pesos_ajustados = {dec: 1 - (dist / dist_total) for dec, dist in distancias.items()}

st.write("### Pesos Ajustados (aₖ)")
st.dataframe(pd.DataFrame({
    'Decisor': list(pesos_ajustados.keys()),
    'Peso Ajustado': list(pesos_ajustados.values())
}))

# === Etapa 5: Vetor Combinado BDMM ===
st.header("5. Vetor de Pesos Combinado do BDMM")
st.markdown(r"""
O vetor combinado é:

$$
W_{comb,l} = \sum_{k=1}^{n} a_k \cdot w'_{kl}
$$

Após isso, normalizamos para garantir:

$$
\sum_l W_{comb,l} = 1
$$
""")

df_pesos['Peso_Combinado'] = 0

for i in range(num_decisores):
    decisor = f'D{i+1}'
    df_pesos['Peso_Combinado'] += pesos_ajustados[decisor] * df_pesos[decisor]

# Normalização final (CORREÇÃO CRUCIAL)
df_pesos['Peso_Combinado'] = df_pesos['Peso_Combinado'] / df_pesos['Peso_Combinado'].sum()

st.write("### Pesos Combinados Normalizados")
st.dataframe(df_pesos[['Critério', 'Peso_Combinado']])

# === Etapa 6: Índice de Consenso ===
st.header("6. Índice de Consenso (CI)")

colunas = [f'D{i+1}' for i in range(num_decisores)]
desvio = df_pesos[colunas].std(axis=1)
media = df_pesos[colunas].mean(axis=1)

# sigma_max corrigido
desvio_max = np.sqrt(media * (1 - media))
CI = 1 - (desvio / desvio_max.replace(0, np.nan))
CI = CI.fillna(1).clip(upper=1)

def classificar_ci(x):
    if x >= 0.85: return 'Alto Consenso'
    if x >= 0.70: return 'Moderado'
    if x >= 0.50: return 'Baixo'
    return 'Dissenso'

df_consenso = pd.DataFrame({
    'Critério': df_pesos['Critério'],
    'Desvio-Padrão': desvio,
    'Índice de Consenso (CI)': CI
})

df_consenso['Nível de Consenso'] = df_consenso['Índice de Consenso (CI)'].apply(classificar_ci)

st.write("### Índice de Consenso por Critério")
st.dataframe(df_consenso)

# === Etapa 7: Visualização ===
st.header("7. Visualização")
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(df_pesos['Critério'], df_pesos['Peso_Combinado'])
ax.set_ylabel("Peso Combinado")
ax.set_title("Pesos Combinados Finais (BDMM)")
st.pyplot(fig)

