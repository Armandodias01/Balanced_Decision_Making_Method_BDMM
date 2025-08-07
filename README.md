📊 Balanced Decision-Making Method (BDMM)
O BDMM (Balanced Decision-Making Method) é uma abordagem que busca balancear diferentes conjuntos de pesos (critérios) de tomadores de decisão utilizando distância euclidiana em relação à distribuição uniforme. Essa técnica é útil em contextos de análise multicritério (MCDA) quando é necessário agregar pesos de diferentes perfis de preferência (por exemplo, P1 e P2), equilibrando a influência de cada perspectiva.

✨ Funcionalidades
1 - Calcula distâncias euclidianas dos vetores de pesos (P1 e P2) à distribuição de pesos iguais;

2 - Normaliza as distâncias para obter pesos relativos entre os perfis de decisão;

3 - Gera uma ponderação combinada (W_comb) que equilibra os pesos individuais conforme a similaridade com a neutralidade;

4 - Suporta aplicações em modelos MCDA como AHP, TOPSIS, PROMETHEE, etc.

🧠 Lógica do Método
1 - Entrada: Pesos dos critérios definidos por dois perfis decisórios (P1 e P2).

2 - Cálculo da distribuição neutra: Assumindo pesos iguais para todos os critérios.

3 - Medição de distâncias: Usa distância euclidiana para mensurar quão distantes P1 e P2 estão da neutralidade.

4 - Normalização das distâncias: Transforma as distâncias em proporções relativas.

5 - Ponderação balanceada: Combina os pesos de P1 e P2 considerando o quanto cada vetor se aproxima da neutralidade.
