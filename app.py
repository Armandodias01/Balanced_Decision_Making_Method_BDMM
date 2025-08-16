import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Balanced Decision-Making Method (BDMM) com Índice de Consenso")

st.markdown("""
O **BDMM (Balanced Decision-Making Method)** combina os pesos de múltiplos decisores com base na distância de seus vetores de pesos individuais em relação a um **vetor de pesos iguais**, garantindo que nenhum critério seja priorizado inicialmente.

As etapas principais são:

1. Calcular a distância Euclidiana de cada vetor de pesos individual em relação ao vetor de pesos iguais.
2. Normalizar as distâncias para obter comparabilidade.
3. Ajustar os pesos com base na proximidade ao vetor igual.
4. Combinar os vetores individuais ponderados pelos pesos ajustados.
5. Avaliar o **índice de consenso (CI)** para medir concordância entre decisores.
""")

# === Etapa 1: Entrada de dados ===
st.header("1. Entrada de Dados")
num_decisores = st.number_input("Número de decisores:", min_value=1, value=2, step=1)
num_criterios = st.number_input("Número de critérios:", min_value=1, value=3, step=1)

st.markdown("Defina os nomes dos critérios e os pesos atribuídos por cada decisor (de 0 a 1, cada decisor deve somar 1).")
nomes_criterios = []
for i in range(num_criterios):
    criterio = st.text_input(f"Nome do critério {i+1}:", value=f"C{i+1}")
    nomes_criterios.append(criterio)

dados_pesos = {'Critério': nomes_criterios}
for i in range(num_decisores):
    decisor = f'D{i+1}'
    pesos = []
    st.markdown(f"**Decisor {decisor}**")
    for j in range(num_criterios):
        peso = st.number_input(
            f"Peso para {nomes_criterios[j]} ({decisor}):",
            min_value=0.0, max_value=1.0, value=1/num_criterios, step=0.01, key=f"{decisor}_{j}"
        )
        pesos.append(peso)
    dados_pesos[decisor] = pesos

df_pesos = pd.DataFrame(dados_pesos)

# Normalização automática
for i in range(num_decisores):
    decisor = f'D{i+1}'
    soma = df_pesos[decisor].sum()
    if not np.isclose(soma, 1.0):
        df_pesos[decisor] = df_pesos[decisor] / soma

st.write("### Pesos normalizados por decisor")
st.dataframe(df_pesos)

# === Etapa 2: Vetor de pesos iguais ===
st.header("2. Vetor de Pesos Iguais")
n = len(df_pesos['Critério'])
df_pesos['Pesos_Iguais'] = [1/n] * n
st.markdown(f"O vetor de pesos iguais é definido como 1/{n} para cada critério, garantindo neutralidade inicial.")

# === Etapa 3: Distância Euclidiana ===
st.header("3. Distância Euclidiana de cada decisor")
st.markdown("""
A distância Euclidiana de cada vetor de pesos individual Wk em relação ao vetor de pesos iguais Weq é calculada por:

$$
d_k = \\sqrt{\\sum_{l=1}^{m} (w_{kl} - w_{eq,l})^2}
$$

Onde:
$$
- \(w_{kl}\) é o peso do critério l do decisor k
$$
$$
- \(w_{eq,l} = 1/m\)
$$
$$
- m = número de critérios
$$
""")

distancias_decisores = {}
distancia_total = 0
for i in range(num_decisores):
    decisor = f'D{i+1}'
    df_pesos[f'{decisor}_Diff'] = (df_pesos[decisor] - df_pesos['Pesos_Iguais'])**2
    distancia = np.sqrt(df_pesos[f'{decisor}_Diff'].sum())
    distancias_decisores[decisor] = distancia
    distancia_total += distancia

# === Etapa 4: Normalização das distâncias e pesos ajustados ===
st.header("4. Normalização das Distâncias e Ajuste dos Pesos")
st.markdown("""
A distância normalizada de cada decisor é calculada como:

$$
d_k^{norm} = \\frac{d_k}{\\sum_{i=1}^{m} d_i}
$$

O **peso ajustado** de cada vetor é dado por:

$$
w_k^{adj} = 1 - d_k^{norm}
$$

Vetores mais próximos do vetor igual recebem maior peso ajustado.
""")

pesos_normalizados = {f'Normalizado_{decisor}': dist/distancia_total for decisor, dist in distancias_decisores.items()}
pesos_ajustados = {dec: 1 - norm for dec, norm in pesos_normalizados.items()}

st.write("### Pesos ajustados dos decisores")
df_pesos_ajustados = pd.DataFrame({
    'Decisor': list(pesos_ajustados.keys()),
    'Peso Ajustado': list(pesos_ajustados.values())
})
st.dataframe(df_pesos_ajustados)

# === Etapa 5: Cálculo dos Pesos Combinados ===
st.header("5. Pesos Combinados Finais")
st.markdown("""
O vetor combinado Wcomb é obtido como a média ponderada de todos os vetores de decisores usando os pesos ajustados:

$$
W_{comb} = \\sum_{k=1}^{m} (w_k^{adj} \\cdot W_k)
$$
""")

df_pesos['Peso_Combinado'] = 0
for i in range(num_decisores):
    decisor = f'D{i+1}'
    nome_norm = f'Normalizado_{decisor}'
    df_pesos['Peso_Combinado'] += pesos_ajustados[nome_norm] * df_pesos[decisor]

st.write("### Vetor de Pesos Combinados")
st.dataframe(df_pesos[['Critério', 'Peso_Combinado']])

# === Etapa 6: Índice de Consenso ===
st.header("6. Índice de Consenso (CI)")
st.markdown("""
O **Índice de Consenso (CI)** mede a concordância entre decisores:

$$
CI = 1 - \\frac{\\sigma_{obs}}{\\sigma_{max}}
$$

- \(\\sigma_{obs}\) = desvio-padrão dos pesos de cada critério
- \(\\sigma_{max} = \\sqrt{\\mu (1 - \\mu)}\), onde \(\\mu\) é a média dos pesos
- CI próximo de 1 indica alto consenso, CI próximo de 0 indica divergência.
""")

colunas_decisores = [f'D{i+1}' for i in range(num_decisores)]
desvio = df_pesos[colunas_decisores].std(axis=1, ddof=1)
media = df_pesos[colunas_decisores].mean(axis=1)
desvio_max = np.sqrt(media * (1 - media))
desvio_max_seguro = desvio_max.replace(0, np.nan)
CI = 1 - (desvio / desvio_max_seguro).clip(upper=1)

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

st.write("### Índice de Consenso por Critério")
st.dataframe(df_consenso)

# === Etapa 7: Visualização ===
st.header("7. Visualização dos Pesos Combinados")
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(df_pesos['Critério'], df_pesos['Peso_Combinado'], color='steelblue')
ax.set_ylabel("Peso Combinado")
ax.set_title("Pesos Combinados Finais (BDMM)")
st.pyplot(fig)
