import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Balanced Decision-Making Method (BDMM) com Índice de Consenso")

st.markdown("""
Este aplicativo implementa o **Método de Decisão Balanceada (BDMM)**, que combina pesos de múltiplos decisores para gerar uma ponderação final para cada critério.  
Também calcula o **Índice de Consenso (CI)**, que mede o quanto os decisores concordam entre si.
""")

# === Etapa 1: Entrada de dados ===
st.header("1. Entrada de Dados")
num_decisores = st.number_input("Número de decisores:", min_value=1, value=2, step=1)
num_criterios = st.number_input("Número de critérios:", min_value=1, value=3, step=1)

st.markdown("Defina os nomes dos critérios e os pesos atribuídos por cada decisor (de 0 a 1, somando 1 para cada decisor).")
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

# Normalização automática dos pesos caso a soma não seja 1
for i in range(num_decisores):
    decisor = f'D{i+1}'
    soma = df_pesos[decisor].sum()
    if not np.isclose(soma, 1.0):
        df_pesos[decisor] = df_pesos[decisor] / soma

st.write("### Pesos normalizados por decisor")
st.dataframe(df_pesos)

# === Etapa 2: Pesos iguais ===
st.header("2. Definição dos Pesos Iguais")
n = len(df_pesos['Critério'])
df_pesos['Pesos_Iguais'] = [1/n] * n
st.markdown(f"Aqui definimos pesos iguais para todos os critérios: {1/n:.3f} para cada um.")

# === Etapa 3: Distâncias Euclidianas ===
st.header("3. Cálculo das Distâncias Euclidianas")
st.markdown("""
Calculamos a **distância Euclidiana** de cada vetor de pesos de decisor em relação ao vetor de pesos iguais.  
Isso nos mostra o quanto cada decisor se afasta de uma ponderação "balanceada".
""")

distancias_decisores = {}
distancia_total = 0
for i in range(num_decisores):
    decisor = f'D{i+1}'
    df_pesos[f'{decisor}_Diff'] = (df_pesos[decisor] - df_pesos['Pesos_Iguais'])**2
    distancia = np.sqrt(df_pesos[f'{decisor}_Diff'].sum())
    distancias_decisores[decisor] = distancia
    distancia_total += distancia

# Normalização das distâncias para ponderação combinada
pesos_normalizados = {f'Normalizado_{decisor}': dist/distancia_total for decisor, dist in distancias_decisores.items()}

st.write("### Distâncias e pesos normalizados")
df_resultados = pd.DataFrame({
    'Métrica': [f'{dec}_Distância' for dec in distancias_decisores.keys()] + list(pesos_normalizados.keys()),
    'Valor': list(distancias_decisores.values()) + list(pesos_normalizados.values())
})
st.dataframe(df_resultados)

# === Etapa 4: Pesos Combinados ===
st.header("4. Cálculo dos Pesos Combinados")
st.markdown("""
Os **pesos combinados** são obtidos usando os pesos normalizados das distâncias como fatores de ponderação para cada decisor:

$$
W_{combinado} = \\sum_{i=1}^{m} (w_i^{normalizado} \\cdot w_i^{decisor})
$$

onde $m$ é o número de decisores.
""")

df_pesos['Peso_Combinado'] = 0
for i in range(num_decisores):
    decisor = f'D{i+1}'
    nome_norm = f'Normalizado_{decisor}'
    df_pesos['Peso_Combinado'] += pesos_normalizados[nome_norm] * df_pesos[decisor]

st.write("### Pesos Combinados por Critério")
st.dataframe(df_pesos[['Critério', 'Peso_Combinado']])

# === Etapa 5: Índice de Consenso ===
st.header("5. Índice de Consenso (CI)")
st.markdown("""
O **Índice de Consenso** avalia o quanto os decisores concordam na atribuição de pesos.  
Ele é baseado no **desvio-padrão dos pesos** de cada critério:

$$
CI = 1 - \\frac{\\sigma_{obs}}{\\sigma_{max}}
$$

- $\\sigma_{obs}$: desvio-padrão observado dos pesos de cada critério  
- $\\sigma_{max} = \\sqrt{\\mu \\cdot (1 - \\mu)}$: desvio máximo teórico  
- $\\mu$: média dos pesos atribuídos pelos decisores
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

st.markdown("""
**Interpretação do CI:**

- $CI \\ge 0.85$: Alto consenso  
- $0.70 \\le CI < 0.85$: Moderado  
- $0.50 \\le CI < 0.70$: Baixo  
- $CI < 0.50$: Dissenso

Um CI próximo de 1 indica forte concordância entre os decisores, enquanto valores próximos de 0 indicam divergência.
""")

# === Etapa 6: Visualização dos Pesos Combinados ===
st.header("6. Visualização Gráfica")
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(df_pesos['Critério'], df_pesos['Peso_Combinado'], color='steelblue')
ax.set_ylabel("Peso Combinado")
ax.set_title("Pesos Combinados Finais por Critério (BDMM)")
st.pyplot(fig)
