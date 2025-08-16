import streamlit as st
import pandas as pd
import numpy as np

st.title("BDMM - Medida de Consenso via Desvio-Padrão")

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

# === Cálculo do desvio-padrão por critério ===
colunas_decisores = [f'D{i+1}' for i in range(num_decisores)]
desvio = df_pesos[colunas_decisores].std(axis=1, ddof=1)

# Média por critério
media = df_pesos[colunas_decisores].mean(axis=1)

# Desvio máximo teórico e dispersão relativa
desvio_max = np.sqrt(media * (1 - media))
desvio_max_seguro = desvio_max.replace(0, np.nan)
disp_relativa = (desvio / desvio_max_seguro).clip(upper=1)
CI = 1 - disp_relativa

# Classificação do nível de consenso
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

st.write("### Resultado do Consenso por Critério")
st.dataframe(df_consenso)

# Mostrar legenda das faixas
st.markdown("""
**Interpretação do índice de consenso (CI):**

- **CI ≥ 0,85:** Alto consenso  
- **0,70 ≤ CI < 0,85:** Moderado  
- **0,50 ≤ CI < 0,70:** Baixo  
- **CI < 0,50:** Dissenso
""")
