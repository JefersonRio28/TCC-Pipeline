# =========================================================================================================================
# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# TRABALHO DE CONCLUSÃO DE CURSO (TCC)
# Aluno: Jeferson Felipe Rio Vicente
#
# Descrição:
#      1 - Processamento e manipulação de 3 bases de dados (partidas, jogadores e equipes), integrando tudo
#          em uma tabela "macro"
#      2 - Análise exploratória das variáveis (Matriz de correlação)
#      3 - Algoritmo Random Forest tentando prever o vencedor da partida, extraindo o peso estatístico
#          dos fundamentos (aces, bloqueios, ataques, ...)
#      4 - Agrupamento (Clustering) de jogadores com base nos pesos obtidos no algoritmo de predição.
#      5 - Descrição dos resultados
#
# ========================================================================================================================
# In[0]: Importação das bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, learning_curve, validation_curve, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, silhouette_score, davies_bouldin_score, calinski_harabasz_score, confusion_matrix
from sklearn.inspection import permutation_importance

# In[1]: Data loading and integration

print("1 - Carregando e integrando as bases de dados...")

df_matches = pd.read_csv('matchStats.csv') # Placar, números dos fundamentos técnicos (Mandante e Visitante)
df_players = pd.read_csv('playerStats.csv') # Informações dos jogadores (estatísticas na VNL)
df_teams = pd.read_csv('teamStats.csv') # Histórico de cada seleção nacional

# print(df_matches.info())
# print(df_players.info())
# print(df_teams.info())

data_lines = [] # Inicialização da tabela macro

# Percorre o dataframe para calcular a proporção de pontos (Equipe/Pontos Totais) de cada equipe com base no jogo (linha)
for index, line in df_matches.iterrows():
    total_points = line['Total Points Home'] + line['Total Points Away'] # Obtendo os pontos totais
    
    # Verificação de Erro - ignora dados corrompidos ou incompletos
    if total_points == 0:
        continue
    
    # Obtendo os pontos das equipes Mandante e Visitante
    stats_home = df_teams[df_teams['Team'] == line ['Home Team']]
    stats_away = df_teams[df_teams['Team'] == line ['Away Team']]
    
    # Verificação de erro
    if stats_home.empty or stats_away.empty:
        continue
    
    # Extraindo o valor numérico de cada equipe
    pr_home = stats_home['Point Ratio'].values[0]
    pr_away = stats_away['Point Ratio'].values[0]
    
    # Medindo a força de cada equipe
    teams_dif = pr_home - pr_away
    
    # Informações da equipe mandante
    home_winner = 1 if line['Winner'] == line['Home Team'] else 0
    data_lines.append({
        'ID_Jogo': index,
        'Ganhador': home_winner,
        'Taxa_Aces': line['Aces Home']/total_points,
        'Taxa_Blocks': line['Blocks Home']/total_points,
        'Taxa_Ataques': line['Kills Home']/total_points,
        'Taxa_Defesas': line['Digs Home']/total_points,
        'Erros_adversarios': line['Opponents Errors Away']/total_points,
        'Dif_Times': teams_dif,
    })
    
    # O ID_Match manterá as observações juntas no treino e teste. A Proporção quantifica o rastreamento dos dados
    # padronizando-os com base no total_points
    
    # Informações da equipe visitante
    away_winner = 1 if line['Winner'] == line['Away Team'] else 0
    data_lines.append({
        'ID_Jogo': index,
        'Ganhador': away_winner,
        'Taxa_Aces': line['Aces Away']/total_points,
        'Taxa_Blocks': line['Blocks Away']/total_points,
        'Taxa_Ataques': line['Kills Away']/total_points,
        'Taxa_Defesas': line['Digs Away']/total_points,
        'Erros_adversarios': line['Opponents Errors Home']/total_points,
        'Dif_Times': -teams_dif,
    })
    
df_macro = pd.DataFrame(data_lines)
attributes_list = ['Taxa_Aces', 'Taxa_Blocks', 'Taxa_Ataques', 'Taxa_Defesas', 'Erros_adversarios', 'Dif_Times']

# In[2]: Verificação de correlação

print('2 - Gerando a matriz de correlação de Pearson...')

# Analisando a matriz de correlação como um mapa de calor (heatmap)
corr_matrix = df_macro[attributes_list].corr()

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax)
#ax.set_title('Matriz de Correlação de Pearson: Validação Multivariada', pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_correlacao.png', dpi=300)
plt.close()

# In[3a]: Modelagem Supervisionada (Random Forest)

#Treino de um classificador que prevê o vencedor (1) a partir das 6 taxas técnicas + diferença de pontos, obtendo no fim as métricas
#de desempenho e o peso relativo de cada fundamento na vitória

print('3a - Treinando Random Forest e extraindo as métricas de desempenho')

X = df_macro[attributes_list] # Variáveis explicativas
y = df_macro['Ganhador'] # Variavel dependente
groups = df_macro['ID_Jogo'] # Orientação do split entre conjunto ganhador/perdedor

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # 80% Treino / 20% Teste

index_training, index_test = next(splitter.split(X, y, groups=groups))
X_training, X_test = X.iloc[index_training], X.iloc[index_test]
y_training, y_test = y.iloc[index_training], y.iloc[index_test]

# Treina o Random Forest com 100 árvores de decisão
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_training, y_training)

pred_test = rf_model.predict(X_test) # Previsão de classe (0 ou 1) para o teste
probab_test = rf_model.predict_proba(X_test)[:, 1] # Probabilidade estimada de vitória

accuracy = accuracy_score(y_test, pred_test) # % de acertos no conjunto de teste
auc = roc_auc_score(y_test, probab_test) # Capacidade de separar vitoria/derrota

#print(f"Acurácia: {accuracy:.3f}")
#print(f"AUC: {auc:.3f}")


# Checagem da inclinação do modelo a estar enviasado para vitórias e derrotas
#print(classification_report(y_test, pred_test, target_names=['Derrota', 'Vitoria']))

# Validação cruzada: medindo a estabilidade dos resultados, não somente para um split aleatório
# Divisão do dataset em grupos para validação
gkf = GroupKFold(n_splits=5)
# Automatização do processo de treino para as 5 Folds
cv_scores = cross_val_score(rf_model, X, y, groups=groups, cv=gkf, scoring='accuracy')
#print(
#      f"Acurácia CV 5-fold (agrupada por partida): {np.round(cv_scores, 3)} | media: {cv_scores.mean():.3f} | desvio: {cv_scores.std():.3f}"
#      )

# Calculo do vetor de Importância de Gini, mensurando quanto cada variável contribuiu para a decisão das árvores
gini_importances = rf_model.feature_importances_
attribute_labels = ['Taxa de Saques', 'Taxa de Bloqueios', 'Taxa de Ataques', 'Taxa de Defesas', 'Erros_adversarios', 'Dif_Times']

# Ordenando os indices de menor para maior importância
sorted_index = np.argsort(gini_importances)
sorted_gini_importances = gini_importances[sorted_index]
sorted_labels = [attribute_labels[i] for i in sorted_index]

# Geração do gráfico de barras horizontais que compara a contribuição de cada fundamento
fig, ax = plt.subplots(figsize=(10, 5))
colors_esalq = ['indigo', 'deeppink', 'goldenrod', 'darkorange', 'teal', 'navy'][:len(sorted_gini_importances)]
bars = ax.barh(sorted_labels, sorted_gini_importances, color=colors_esalq)
ax.bar_label(bars, fmt='%.3f', label_type='center', color='white', fontweight='bold', fontsize=10)
#ax.set_title('Importância Preditiva dos Fundamentos e Consistência Macro', pad=15, fontweight='bold')
ax.set_xlabel('Gini Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# Verificação da efetividade do modelo com os dados sendo apresentados de forma diferente
perm = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)
# n_repeats=30: repete o embaralhamento 30 vezes por variável e tira a média,
# Para o resultado não depender de uma única permutação.
importance_perm = perm.importances_mean
sorted_index_perm = np.argsort(importance_perm)

fig, ax = plt.subplots(figsize=(10, 5))
colors_perm = ['indigo', 'deeppink', 'goldenrod', 'darkorange', 'teal', 'navy'][:len(importance_perm)]
bars_perm = ax.barh([attribute_labels[i] for i in sorted_index_perm], importance_perm[sorted_index_perm], color=colors_perm)
ax.bar_label(bars_perm, fmt='%.3f', label_type='center', color='white', fontweight='bold', fontsize=10)
#ax.set_title('Importância por Permutação (checagem de robustez do Score de Gini)', pad=15, fontweight='bold')
ax.set_xlabel('Redução Média de Acurácia na Permutação')
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=300)
plt.close()

# In[3b]: Testando o efeito da quantidade de dados no treino e a curva de aprendizado

#Duas abordagens complementares:
#   1 - Comparação direta: repetição do treino/avaliação para várias proporções de teste (10% a 50%), com várias divisões aleatórias
#3       em cada proporção, separando o efeito da proporção do efeito de "sorte" de um único sorteio de linhas
#   2 - Curva de Aprendizado: Variação do tamanho absoluto do conjunto de treino (com validação cruzada em cada tamanho), mostrando como a
#       acurácia evolui, olhando para indicios de overfitting e verificar o se mais dados tenderiam a ajudar

print("3b - Comparando proporções de treino/teste e curva de aprendizado")

test_proportion = [0.1, 0.2, 0.3, 0.4, 0.5]
n_repetitions = 30
split_results = []

for test in test_proportion:
    accs_rep, aucs_rep = [], []
    for rep in range(n_repetitions):
        """ random_state=rep: cada repetição usa uma divisão diferente das partidas em treino/teste (sempre respeitando o agrupamento por
        partida, como na Etapa 3 principal)."""
        splitter_rep = GroupShuffleSplit(n_splits=1, test_size=test, random_state=rep)
        itr, ite = next(splitter_rep.split(X, y, groups=groups))
        model_rep = RandomForestClassifier(n_estimators=100, random_state=42)
        model_rep.fit(X.iloc[itr], y.iloc[itr])
        pred_rep = model_rep.predict(X.iloc[ite])
        proba_rep = model_rep.predict_proba(X.iloc[ite])[:, 1]
        accs_rep.append(accuracy_score(y.iloc[ite], pred_rep))
        aucs_rep.append(roc_auc_score(y.iloc[ite], proba_rep))
    split_results.append({
        'test_size': test, 'n_treino': len(itr), 'n_teste': len(ite),
        'acc_media': np.mean(accs_rep), 'acc_std': np.std(accs_rep),
        'auc_media': np.mean(aucs_rep), 'auc_std': np.std(aucs_rep),
    })

df_split = pd.DataFrame(split_results)
#print(df_split.round(3).to_string(index=False))

# Gráfico com 2 painéis: acurácia e AUC em função da % reservada para teste,
# com barra de erro mostrando o desvio-padrão entre as 30 repetições de cada
# proporção.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.errorbar(df_split['test_size'] * 100, df_split['acc_media'], yerr=df_split['acc_std'],
             marker='o', color='indigo', capsize=4, linewidth=2)
ax1.axvline(x=20, color='deeppink', linestyle='--', linewidth=1.5, label='Proporção adotada (80/20)')
ax1.set_xlabel('% dados no teste')
ax1.set_ylabel('Acurácia média')
#ax1.set_title('Acurácia por proporção de teste')
ax1.legend()

ax2.errorbar(df_split['test_size'] * 100, df_split['auc_media'], yerr=df_split['auc_std'],
             marker='o', color='teal', capsize=4, linewidth=2)
ax2.axvline(x=20, color='deeppink', linestyle='--', linewidth=1.5, label='Proporção adotada (80/20)')
ax2.set_xlabel('% dados no teste')
ax2.set_ylabel('AUC média')
#ax2.set_title('AUC por proporção de teste')
ax2.legend()

plt.tight_layout()
plt.savefig('comparacao_splits.png', dpi=300)
plt.close()

# Cruva de aprendizado: rodando uma validação cruzada de 5 folds, devolvendo a acurácia de treino e a validação em cada fold baseado
# no score_treino/score_validação.

training_size, training_scores, scores_validation = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, groups=groups, cv=GroupKFold(n_splits=5),
    train_sizes=np.linspace(0.2, 1.0, 6), scoring='accuracy'
)

mean_training, std_training = training_scores.mean(axis=1), training_scores.std(axis=1)
mean_validation, std_validation = scores_validation.mean(axis=1), scores_validation.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(training_size, mean_training, marker='o', color='navy', linewidth=2, label='Acurácia no treino')
ax.fill_between(training_size, mean_training - std_training, mean_training + std_training, color='navy', alpha=0.15)
ax.plot(training_size, mean_validation, marker='o', color='deeppink', linewidth=2, label='Acurácia na validação (CV agrupada, 5-fold)')
ax.fill_between(training_size, mean_validation - std_validation, mean_validation + std_validation, color='deeppink', alpha=0.15)
ax.set_xlabel('Número de observações no treino')
ax.set_ylabel('Acurácia')
#ax.set_title('Curva de Aprendizado: efeito do tamanho do treino no desempenho')
ax.legend()
plt.tight_layout()
plt.savefig('curva_aprendizado.png', dpi=300)
plt.close()

# In[3c]: Testando o efeito do número de árvores (n_estimators)

print("3c - Avaliando o efeito do número de árvores na previsão")

n_estimators_values = [10, 25, 50, 75, 100, 150, 200, 300, 500]

scores_training_ne, scores_validation_ne = validation_curve(
    RandomForestClassifier(random_state=42), X, y, param_name='n_estimators',
    param_range=n_estimators_values, groups=groups, cv=GroupKFold(n_splits=5), scoring='accuracy'
)
mean_training_ne = scores_training_ne.mean(axis=1)
mean_validation_ne = scores_validation_ne.mean(axis=1)
std_validation_ne = scores_validation_ne.std(axis=1)

# print(pd.DataFrame({
#     'n_estimators': n_estimators_values,
#     'acc_treino': mean_training_ne.round(3),
#     'acc_val_media': mean_validation_ne.round(3),
#     'acc_val_std': std_validation_ne.round(3),
# }).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(n_estimators_values, mean_training_ne, marker='o', color='navy', linewidth=2, label='Acurácia no treino')
ax.plot(n_estimators_values, mean_validation_ne, marker='o', color='deeppink', linewidth=2, label='Acurácia na validação (CV agrupada, 5-fold)')
ax.fill_between(n_estimators_values, mean_validation_ne - std_validation_ne, mean_validation_ne + std_validation_ne, color='deeppink', alpha=0.15)
ax.axvline(x=100, color='indigo', linestyle='--', linewidth=1.5, label='n_estimators adotado (100)')
ax.set_xlabel('Número de árvores (n_estimators)')
ax.set_ylabel('Acurácia')
#ax.set_title('Sensibilidade da Acurácia ao Número de Árvores do Random Forest')
ax.legend()
plt.tight_layout()
plt.savefig('validacao_n_estimators.png', dpi=300)
plt.close()

importances_per_n = np.array([
    RandomForestClassifier(n_estimators=n, random_state=42).fit(X, y).feature_importances_
    for n in n_estimators_values
])  # formato: (número de valores de n_estimators testados) x (6 variáveis)

fig, ax = plt.subplots(figsize=(10, 5.5))
colors_attributes = ['indigo', 'deeppink', 'goldenrod', 'darkorange', 'teal', 'navy']
for j, label in enumerate(attribute_labels):
    ax.plot(n_estimators_values, importances_per_n[:, j], marker='o', color=colors_attributes[j], linewidth=2, label=label)
ax.axvline(x=100, color='gray', linestyle='--', linewidth=1.5, label='n_estimators adotado (100)')
ax.set_xlabel('Número de árvores (n_estimators)')
ax.set_ylabel('Gini Score')
#ax.set_title('Estabilidade do Ranking de Importância conforme o Número de Árvores')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('estabilidade_importancia_n_estimators.png', dpi=300)
plt.close()

# In[4]: Definição da população alvo

print("4 - Tratando a base de dados para definição dos jogadores a serem avaliados")

# Mudança dos dados dos jogadores para anonimizar eles
df_players['ID_Atleta'] = [f"Atleta_{i+1:03d}" for i in range(len(df_players))]
df_players = df_players.drop(columns=['Player Name'])

# Definindo os parâmetros de forma a avaliar a eficiência de cada fundamento baseada no total de tentativas
df_players['Total_Saques'] = df_players['Aces'] + df_players['Service Errors'] + df_players['Service Attempts']
df_players['Total_Ataques'] = df_players['Kills'] + df_players['Attacking Errors'] + df_players['Attacking Attempts']

df_players['Taxa_Aces_Jogador'] = np.where(df_players['Total_Saques'] > 0, df_players['Aces'] / df_players['Total_Saques'], 0)
df_players['Taxa_Bloqueios_Jogador'] = df_players['Blocks Per Match']
df_players['Taxa_Ataques_Jogador'] = np.where(df_players['Total_Ataques'] > 0, (df_players['Kills'] - df_players['Attacking Errors']) / df_players['Total_Ataques'], 0)
df_players['Taxa_Defesas_Jogador'] = df_players['Digs Per Match']

player_attributes = ['Taxa_Aces_Jogador', 'Taxa_Bloqueios_Jogador', 'Taxa_Ataques_Jogador', 'Taxa_Defesas_Jogador']

# As posições tem de ser delimitadas por conta de indicadores especificos presentes em cada posição. Como levantadores e liberos tem funções
# distintas desses jogadores eles precisam ser excluidos de forma a melhorar a análise baseada em uma população mas especifica
net_positions = ['OUTSIDE HITTER', 'MIDDLE BLOCKER', 'OPPOSITE SPIKER']

total_number_players = len(df_players)

df_net_players = df_players[df_players['Position'].isin(net_positions)].copy()

total_number_net_players = len(df_net_players)

players_analysis_ratio = total_number_net_players/total_number_players*100
#print(f"Cerca de {players_analysis_ratio}% estão sendo representados nessa base de dados")

# Limpeza dos dados por conta de outliers quando os valores são influenciados diretamente pela questão de acerto/tentativa
df_players_cleaned = df_net_players[(df_net_players['Total_Ataques'] > 10) & (df_net_players['Total_Saques'] > 10)].copy()
#print(f"Jogadores após filtro (volume mínimo de ataques e saques): {len(df_players_cleaned)} de {len(df_players)}")


# In[5]: Clusterização por K-Means e determinação do número de clusters por diferentes técnicas

print("5 - Executando os testes para determinação do número de clusters")

# Padronização das variáveis pelo Z-score
normalize = StandardScaler()
X_scalled_player = normalize.fit_transform(df_players_cleaned[player_attributes])

# A partir da análise Random Forest aqui os pesos são atribuidos a cada fundamento para a definição dos clusters
rf_weights = gini_importances[:4]
X_players_weighed = X_scalled_player * rf_weights

# Definição do intervalo de variação dos clusters (1 a 10), acompanhado pelo vetor que recebera a Soma dos Quadrados Intra-Cluster (WCSS),
# a Silhouette (distância entre o jogador e seu cluster), Davies-Bouldin (razão entre os centros dos clusters) e Calinski-Harabasz (razao entre
# a dispersao entre clusters e dentro de cada cluster)
wcss = []
silhouettes = []
davies_bouldies = []
calinski_harabasz = []
clusters_interval = range(1, 11)

for k in clusters_interval:
    # n_init=10: roda o K-Means 10 vezes com centróides iniciais diferentes e
    # fica com o melhor resultado — reduz o risco de cair num mínimo local ruim.
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_test = kmeans_test.fit_predict(X_players_weighed)  # treina e já devolve o cluster de cada jogador
    wcss.append(kmeans_test.inertia_)  # .inertia_ é o WCSS calculado internamente pelo sklearn
    
    # Enquanto o cotovelo é determinado "a olho" a partir do gráfico, os outros métodos fazem comparações internas e com a vizinhança, dessa forma
    # dependem de um número minimo de clusters, ou seja, pelo menos 2.
    if k >= 2:
        silhouettes.append(silhouette_score(X_players_weighed, labels_test))
        davies_bouldies.append(davies_bouldin_score(X_players_weighed, labels_test))
        calinski_harabasz.append(calinski_harabasz_score(X_players_weighed, labels_test))
    else:
        silhouettes.append(np.nan)
        davies_bouldies.append(np.nan)
        calinski_harabasz.append(np.nan)


#Atribuição dos valores dos scores obtidos na variação do número de clusters
db_values = [d for d in davies_bouldies if not np.isnan(d)]
ch_values = [c for c in calinski_harabasz if not np.isnan(c)]
ks_values = [k for k in clusters_interval if k >= 2]
silh_values = [s for s in silhouettes if not np.isnan(s)]

# Detecção automática do número de clusters pelo método do cotovelo sem a necessidade de fazer a avaliação gráfica. Nessa função, é criada uma função
# entre o ponto n e o ponto n+1, que calcula a distância entre eles.
def elbow_detect(ks, values):
    ks = np.array(ks, dtype=float)
    values = np.array(values, dtype=float)
    # Normalização dos dois eixos para 0-1, para que a distância seria dominada pela escala do WCSS (que é bem maior que a escala de K)
    ks_norm = (ks - ks.min()) / (ks.max() - ks.min())
    values_norm = (values - values.min()) / (values.max() - values.min())
    p1, p2 = np.array([ks_norm[0], values_norm[0]]), np.array([ks_norm[-1], values_norm[-1]])
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    distances = []
    for x, y in zip(ks_norm, values_norm):
        vector = np.array([x, y]) - p1
        projection = np.dot(vector, direction) * direction
        distances.append(np.linalg.norm(vector - projection))  # distância perpendicular à reta
    return distances

all_elbow_distance = elbow_detect(list(clusters_interval), wcss)
k_elbow_auto = clusters_interval[int(np.argmax(all_elbow_distance))]
elbow_distances = [all_elbow_distance[k - 1] for k in ks_values]  # restringe a K>=2, mesmo universo dos outros índices

comparison_table = pd.DataFrame({
    'K': ks_values,
    'Distancia_Cotovelo': elbow_distances,
    'Silhouette': silh_values,
    'Davies_Bouldin': db_values,
    'Calinski_Harabasz': ch_values,
    })

# Conversão dos valores brutos em posições baseadas em cada método
comparison_table['Rank_Cotovelo'] = comparison_table['Distancia_Cotovelo'].rank(ascending=False)
comparison_table['Rank_Silhouette'] = comparison_table['Silhouette'].rank(ascending=False)
comparison_table['Rank_Davies_Bouldin'] = comparison_table['Davies_Bouldin'].rank(ascending=True)
comparison_table['Rank_Calinski_Harabasz'] = comparison_table['Calinski_Harabasz'].rank(ascending=False)

column_rank = ['Rank_Cotovelo', 'Rank_Silhouette', 'Rank_Davies_Bouldin', 'Rank_Calinski_Harabasz']
comparison_table['Ranking_Medio'] = comparison_table[column_rank].mean(axis=1)
comparison_table = comparison_table.sort_values('Ranking_Medio').reset_index(drop=True)


# K RECOMENDADO: o primeiro da tabela já ordenada pelo Ranking Médio.
k_optimum = int(comparison_table.iloc[0]['K'])

#print("\n-> Tabela de comparação multi-critério (ordenada pelo Ranking Médio — menor é melhor):")
#print(comparison_table[['K', 'Distancia_Cotovelo', 'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz', 'Ranking_Medio']].round(3).to_string(index=False))
#print(f"\n-> K recomendado pela combinação dos 4 critérios: K={k_recomendado}")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))
for ax, ks_eixo, valores, titulo, ylabel in [
    (ax1, clusters_interval, wcss, 'Método do Cotovelo (WCSS)', 'Inércia Intra-Cluster (WCSS)'),
    (ax2, ks_values, silh_values, 'Silhouette', 'Silhouette Score'),
    (ax3, ks_values, db_values, 'Davies-Bouldin', 'Davies-Bouldin'),
    (ax4, ks_values, ch_values, 'Calinski-Harabasz', 'Calinski-Harabasz'),
]:
    ax.plot(ks_eixo, valores, marker='o', color='indigo', linewidth=2, markersize=7)
    ax.axvline(x=k_optimum, color='deeppink', linestyle='--', linewidth=1.5, label=f'K recomendado ({k_optimum})')
    ax.set_title(titulo, fontweight='bold')
    #ax.set_xlabel('Quantidade de Clusters (K)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks_eixo)
    ax.legend(fontsize=9)
#plt.suptitle(f'Validação do Número de Clusters — 4 Critérios (recomendação combinada: K={k_optimum})', fontweight='bold', fontsize=15, y=1.00)
plt.tight_layout()
plt.savefig('validacao_criterios_k.png', dpi=300)
plt.close()
#print("-> Gráfico 'validacao_criterios_k.png' gerado (WCSS/cotovelo + Silhouette + Davies-Bouldin + Calinski-Harabasz num único painel).")

# Gráfico do Ranking Médio por K, sendo menor a barra, melhor colocado o K está na média dos 4 critérios. Combinandodo, o que efetivamente decide
# o K usado.

k_values_table = comparison_table.sort_values('K')  # reordena por K só para o eixo X ficar em ordem crescente
fig, ax = plt.subplots(figsize=(9, 5.5))
bar_colors = ['deeppink' if k == k_optimum else 'indigo' for k in k_values_table['K']]
bars = ax.bar(k_values_table['K'].astype(str), k_values_table['Ranking_Medio'], color=bar_colors)
ax.bar_label(bars, fmt='%.2f')
ax.set_xlabel('Quantidade de Clusters (K)')
ax.set_ylabel('Ranking Médio')
#ax.set_title(f'Ranking Combinado dos Critérios de Validação de Cluster\n(recomendação: K={k_optimum})', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('ranking_combinado_k.png', dpi=300)
plt.close()
#print("-> Gráfico 'ranking_combinado_k.png' gerado.")

# Estimação FINAL do K-Means, usando k ótimo decidido pela estrutura de comparação. Sendo a mudança automática, caso a base de dados mude.
kmeans_model = KMeans(n_clusters=k_optimum, random_state=42, n_init=10)
df_players_cleaned['Cluster'] = kmeans_model.fit_predict(X_players_weighed)  # grava o cluster de cada jogador numa nova coluna
#print(f"\n-> Tamanho dos clusters (K={k_optimum}): {df_players_cleaned['Cluster'].value_counts().sort_index().to_dict()}")

# In[6]: Gerando o perfil tecnico dos clusters


clusters_profile = df_players_cleaned.groupby('Cluster')[player_attributes].mean().reset_index()

melted_profiles = pd.melt(clusters_profile, id_vars=['Cluster'], value_vars=player_attributes)

perfil = clusters_profile.set_index('Cluster')
id_rede = perfil['Taxa_Bloqueios_Jogador'].idxmax()   # mais bloqueios -> atacante-bloqueador de rede
id_saque = perfil['Taxa_Aces_Jogador'].idxmax()       # maior eficiência de saque -> sacador agressivo
assert id_rede != id_saque, "Os arquétipos não puderam ser distinguidos pelo perfil"
nomes_clusters = {c: f'{c} - Generalista' for c in perfil.index}
nomes_clusters[id_rede] = f'{id_rede} - Atacante-bloqueador de rede'
nomes_clusters[id_saque] = f'{id_saque} - Sacador agressivo'
melted_profiles['Arquétipo'] = melted_profiles['Cluster'].map(nomes_clusters)
ordem_legenda = [nomes_clusters[c] for c in sorted(nomes_clusters)]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=melted_profiles, x='variable', y='value', hue='Arquétipo',
            hue_order=ordem_legenda, palette='Set2', errorbar=None, ax=ax)
ax.set_xlabel('Indicadores Técnicos de Performance Individual')
ax.set_ylabel('Média do indicador')
ax.set_xticks(range(4))
ax.set_xticklabels(['Eficiência Saque (Aces)', 'Bloqueios por Partida', 'Eficiência Líquida Ataque', 'Defesas por Partida'])
ax.legend(title='Arquétipos Estratégicos', loc='upper left')

# Rótulos nos dados (2 casas decimais, com vírgula)
for container in ax.containers:
    ax.bar_label(container, labels=[f'{v:.2f}'.replace('.', ',') for v in container.datavalues],
                 padding=3, fontsize=9)

# Remove o eixo Y (marcas, valores, linha e grade), mantendo apenas o título
ax.tick_params(axis='y', left=False, labelleft=False)
ax.yaxis.grid(False)
sns.despine(ax=ax, left=True)

min_value = melted_profiles['value'].min()
max_value = melted_profiles['value'].max()
narrow = (max_value - min_value) * 0.1
limite_inferior = 0 if min_value >= 0 else min_value - narrow   # sem folga abaixo de zero se não há valores negativos
ax.set_ylim(limite_inferior, max_value + narrow * 1.5)          # folga extra no topo para os rótulos
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('cluster_jogadores.png', dpi=300)
plt.close()

# Exportando a tabela final: ID anonimizado, equipe, posição, cluster atribuído e as 4 variáveis de performance.
df_players_cleaned[['ID_Atleta', 'Team', 'Position', 'Cluster'] + player_attributes].to_csv('resultados_preliminares_jogadores.csv', index=False)

# In[7]: Exportação dos valores numéricos citados no manuscrito

resumo = []
resumo.append(f"Observações macro: {len(df_macro)} ({df_macro['ID_Jogo'].nunique()} partidas)")
resumo.append(f"Teste agrupado: n={len(y_test)} | acurácia={accuracy:.3f} | AUC={auc:.3f}")
resumo.append("Matriz de confusão (linhas=real, colunas=previsto; 0=Derrota, 1=Vitória):\n" + str(confusion_matrix(y_test, pred_test)))
resumo.append(classification_report(y_test, pred_test, target_names=['Derrota', 'Vitória']))
resumo.append(f"CV 5-fold por dobra: {np.round(cv_scores, 3)} | média={cv_scores.mean():.3f} | desvio={cv_scores.std():.3f}")

pred_oof = cross_val_predict(rf_model, X, y, groups=groups, cv=GroupKFold(n_splits=5))
resumo.append("Previsões fora da dobra (todas as observações):\n" + str(confusion_matrix(y, pred_oof)))
resumo.append(classification_report(y, pred_oof, target_names=['Derrota', 'Vitória']))

resumo.append(f"Atletas: total={total_number_players} | rede={total_number_net_players} | "
              f"após filtro={len(df_players_cleaned)} ({len(df_players_cleaned)/total_number_net_players:.1%} da população-alvo)")
resumo.append(f"K recomendado={k_optimum} | grupos: {df_players_cleaned['Cluster'].value_counts().sort_index().to_dict()}")
resumo.append(f"Pesos do RF no K-Means: {dict(zip(player_attributes, np.round(rf_weights, 3)))}")

with open('resumo_resultados.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(resumo))

comparison_table.round(3).to_csv('tabela_criterios_k.csv', index=False)

perfil = df_players_cleaned.groupby('Cluster')[player_attributes].agg(['mean', 'std'])
perfil['n'] = df_players_cleaned.groupby('Cluster').size()
perfil.round(3).to_csv('tabela_perfil_clusters.csv')

grp = df_players_cleaned.groupby('Cluster')[player_attributes]
(grp.std() / grp.mean()).round(2).to_csv('cv_por_cluster.csv')

pd.crosstab(df_players_cleaned['Cluster'], df_players_cleaned['Position']).to_csv('composicao_cluster_posicao.csv')

print("\n=== ANÁLISE CONCLUÍDA ===")