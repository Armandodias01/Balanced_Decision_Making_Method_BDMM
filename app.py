import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("BDMM - Balanced Decision-Making Method com Índice de Consenso")

st.markdown("""
Este aplicativo implementa o **BDMM (Balanced Decision-Making Method)**, combinando os pesos de múltiplos decisores e calculando o **Índice de Consenso (CI)** para medir a concordância entre eles.
""")

# === Etapa 1: Entrada de dados ===
st.header("1. Entrada de Dados")
st.markdown("""
Cada decisor fornece um vetor de pesos para os critérios.  
Os pesos devem estar entre 0 e 1 e somar 1 para cada decisor.
""")

num_decisores = st.number_input("Número de decisores:", min_value=1, value=2)
num_criterios = st.number_input("Número de critérios:", min_value=1, value=3)

nomes_criterios = []
for i in range(num_criterios):
    crit = st.text_input(f"Nome do critério {i+1}", value=f"C{i+1}")
    nomes_criterios.append(crit)

dados_pesos = {'Critério': nomes_criterios}
for i in range(num_decisores):
    decisor = f'D{i+1}'
    pesos = []
    st.markdown(f"**Decisor {decisor}**")
    for j in range(num_criterios):
        p = st.number_input(f"Peso {nomes_criterios[j]} ({decisor})", min_value=0.0, max_value=1.0, value=1/num_criterios, step=0.01, key=f"{decisor}_{j}")
        pesos.append(p)
    dados_pesos[decisor] = pesos

df_pesos = pd.DataFrame(dados_pesos)

# Normalização dos pesos
for i in range(num_decisores):
    decisor = f'D{i+1}'
    soma = df_pesos[decisor].sum()
    if soma != 1:
        df_pesos[decisor] = df_pesos[decisor] / soma

st.write("### Pesos normalizados por decisor")
st.dataframe(df_pesos)

# === Etapa 2: Vetor de Pesos Iguais ===
st.header("2. Vetor de Pesos Iguais")
m = len(df_pesos['Critério'])
df_pesos['Pesos_Iguais'] = 1/m
st.markdown(r"""
O vetor de pesos iguais é definido como 1/m para cada critério, garantindo neutralidade inicial.

Fórmula:

$$
w_{eq,l} = \frac{1}{m}, \quad l = 1, \dots, m
$$
""")

# === Etapa 3: Distância Euclidiana ===
st.header("3. Distância Euclidiana de cada decisor")
st.markdown(r"""
A distância Euclidiana mede o quanto o vetor de pesos de cada decisor se afasta do vetor de pesos iguais:

$$
d_k = \sqrt{\sum_{l=1}^{m} (w_{kl} - w_{eql})^2}
$$

Onde:
- $w_{kl}$ = peso do critério $l$ pelo decisor $k$  
- $w_{eql}$ = peso igual para o critério $l$  
- $m$ = número de critérios
""")

distancias = {}
for i in range(num_decisores):
    decisor = f'D{i+1}'
    dist = np.sqrt(((df_pesos[decisor] - df_pesos['Pesos_Iguais'])**2).sum())
    distancias[decisor] = dist

# === Etapa 4: Normalização das distâncias e pesos ajustados ===
st.header("4. Normalização das Distâncias e Ajuste dos Pesos")
st.markdown(r"""
As distâncias são normalizadas para permitir comparabilidade entre decisores:

$$
Normalized\,Distance\,for\,w_{k} = \frac{d_k}{\sum_{k=1}^{m} d_k}
$$

O **peso ajustado** é dado por:

$$
w_k^{adj} = 1 - d_k^{norm}
$$

Decisores mais próximos do vetor igual recebem maior peso ajustado.
""")

dist_total = sum(distancias.values())
pesos_ajustados = {dec: 1 - (dist / dist_total) for dec, dist in distancias.items()}

st.write("### Pesos Ajustados")
st.dataframe(pd.DataFrame({'Decisor': list(pesos_ajustados.keys()), 'Peso Ajustado': list(pesos_ajustados.values())}))

# === Etapa 5: Vetor Combinado ===
st.header("5. Vetor de Pesos Combinado")
st.markdown(r"""
O vetor combinado é a média ponderada dos vetores de cada decisor usando os pesos ajustados:

$$
W_{comb} = \sum_{k=1}^{n} (w_k^{adj} \cdot W_k)
$$
""")

df_pesos['Peso_Combinado'] = 0
for i in range(num_decisores):
    decisor = f'D{i+1}'
    df_pesos['Peso_Combinado'] += pesos_ajustados[decisor] * df_pesos[decisor]

st.write("### Vetor de Pesos Combinado")
st.dataframe(df_pesos[['Critério', 'Peso_Combinado']])

# === Etapa 6: Índice de Consenso (CI) ===
st.header("6. Índice de Consenso (CI)")
st.markdown(r"""
O Índice de Consenso mede o grau de concordância entre os decisores para cada critério:

$$
CI = 1 - \frac{\sigma_{obs}}{\sigma_{max}}, \quad \sigma_{max} = \sqrt{\mu \cdot (1-\mu)}
$$

- $\sigma_{obs}$ = desvio-padrão dos pesos do critério entre decisores  
- $\mu$ = média dos pesos do critério  
- CI próximo de 1 indica alto consenso, próximo de 0 indica dissenso
""")

colunas = [f'D{i+1}' for i in range(num_decisores)]
desvio = df_pesos[colunas].std(axis=1, ddof=1)
media = df_pesos[colunas].mean(axis=1)
desvio_max = np.sqrt(media * (1 - media))
CI = 1 - (desvio / desvio_max).clip(upper=1)

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

# === Etapa 7: Visualização dos Pesos Combinados ===
st.header("7. Visualização")
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(df_pesos['Critério'], df_pesos['Peso_Combinado'], color='steelblue')
ax.set_ylabel("Peso Combinado")
ax.set_title("Pesos Combinados Finais (BDMM)")
st.pyplot(fig)
