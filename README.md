# Integração de Machine Learning supervisionado e não supervisionado para análise tática no voleibol de elite

Repositório com o código, as bases e os resultados do Trabalho de Conclusão de Curso apresentado ao MBA em Data Science e Analytics da USP/Esalq.

**Autor:** Jeferson Felipe Rio Vicente
**Orientador:** Erik Miguel de Elias
**Ano:** 2026

---

## Sobre o trabalho

A análise estatística no voleibol costuma parar na descrição de médias de acertos e erros, o que não hierarquiza a contribuição de cada fundamento para a vitória nem orienta o agrupamento de atletas por perfil de rendimento.

Este trabalho implementa um pipeline analítico multinível que conecta os dois níveis: o peso preditivo dos fundamentos, estimado a partir do resultado das partidas por um modelo supervisionado, é transferido como ponderação para o agrupamento de atletas no nível individual. Assim, a segmentação ocorre preferencialmente nas dimensões cuja relevância para o resultado coletivo foi previamente estimada, e não sob pesos arbitrários.

---

## Summary (English)

Statistical analysis in volleyball usually stops at descriptive averages of successful and failed actions, which neither ranks the contribution of each skill to winning nor guides the grouping of athletes by performance profile.

This project implements a multilevel analytical pipeline that links both levels. A Random Forest classifier is trained on match-level data to predict the winner, using a match-grouped train/test split to prevent information leakage between the two mirrored observations of the same game. The resulting feature-importance vector, cross-checked against permutation importance, is then used to weight the standardised individual indicators before K-Means clustering. The number of clusters is selected by combining four internal validation criteria: automatic elbow detection, Silhouette, Davies-Bouldin and Calinski-Harabasz.

The data covers the 2025 men's Volleyball Nations League season and comprises three relational files: matches, teams and players. The full pipeline is contained in `TCC_MBA.py` and reproduces every figure and table reported in the thesis.

---

## Dados

Dados secundários de acesso público, referentes à **temporada 2025 da Volleyball Nations League (VNL), naipe masculino**, competição organizada pela Fédération Internationale de Volleyball.

A base foi obtida a partir de compilação disponibilizada publicamente na plataforma Kaggle:

> HOAG, O. **VNL 2025 Mens**. Kaggle, 2025. Disponível em: https://www.kaggle.com/datasets/owenhoag07/vnl-2025-mens

Trata-se de compilação elaborada por terceiros a partir dos registros da competição, e não de exportação direta da base oficial da federação. Os arquivos foram utilizados na forma em que se encontram, sem alteração dos valores originais.

| Arquivo | Unidade de observação | Registros | Variáveis |
| --- | --- | --- | --- |
| `matchStats.csv` | Confronto | 116 | 31 |
| `teamStats.csv` | Seleção | 18 | 18 |
| `playerStats.csv` | Jogador | 339 | 29 |

Observações sobre a base:

- Os únicos valores ausentes estão nas colunas de quarto e quinto sets, não disputados em partidas encerradas por 3-0 ou 3-1. Essas colunas não integram o conjunto de variáveis explicativas, portanto não há imputação.
- As colunas `Service Attempts` e `Attacking Attempts` seguem o padrão da FIVB e contam apenas as ações que mantiveram a bola em jogo. O total de ações de cada fundamento é a soma de pontos diretos, erros e dessas ações.
- Por se tratar de compilação de terceiros, os valores não passaram por conferência contra a fonte oficial. Eventuais inconsistências de registro na origem se propagam para os resultados.
- Os nomes dos atletas são substituídos por códigos anonimizados logo após o carregamento, antes de qualquer etapa analítica. Nenhum dado pessoalmente identificável circula no pipeline nem nos resultados.

---

## Pipeline

O script executa cinco etapas encadeadas:

1. **Integração das bases.** Une os três repositórios e converte cada fundamento em taxa proporcional ao total de pontos disputados na partida, evitando explicar a vitória pelo próprio placar. Cada partida gera duas observações espelhadas, uma por equipe, totalizando 232 registros.
2. **Validação de colinearidade.** Matriz de correlação de Pearson entre as variáveis explicativas, com limiar de referência em |r| = 0,70.
3. **Modelagem supervisionada.** Random Forest para prever o vencedor, com partição agrupada por partida para evitar vazamento de informação entre as observações espelhadas. Inclui avaliação por acurácia, AUC e validação cruzada agrupada, comparação entre importância de Gini e importância por permutação, e análises de sensibilidade à proporção treino/teste e ao número de estimadores.
4. **Preparação do nível individual.** Delimitação da população-alvo aos atletas de rede (ponteiro, central e oposto), filtro de volume mínimo e padronização por Z-Score.
5. **Agrupamento híbrido.** Multiplicação do espaço padronizado pelo vetor de importância do Random Forest e aplicação do K-Means, com o número de grupos definido por comparação entre quatro critérios: detecção automática do cotovelo, Silhouette, Davies-Bouldin e Calinski-Harabasz.

---

## Como executar

Requisitos: Python 3.9 ou superior.

```bash
git clone <url-do-repositorio>
cd <pasta-do-repositorio>
pip install -r requirements.txt
python TCC_MBA.py
```

O script lê os três arquivos CSV do diretório de trabalho e gera as figuras e a tabela de resultados no mesmo local. A execução completa leva poucos minutos em um computador pessoal.

### Dependências

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Estrutura do repositório

```
.
├── TCC_MBA.py                  # pipeline completo, comentado
├── requirements.txt
├── data/
│   ├── matchStats.csv
│   ├── teamStats.csv
│   └── playerStats.csv
├── figuras/                    # saídas geradas pelo script
└── README.md
```

---

## Saídas geradas

| Arquivo | Conteúdo |
| --- | --- |
| `matriz_correlacao.png` | Matriz de correlação de Pearson entre as variáveis explicativas |
| `feature_importance.png` | Importância das variáveis pelo Gini Score |
| `permutation_importance.png` | Importância por permutação, como checagem de robustez |
| `comparacao_splits.png` | Acurácia e AUC em função da proporção reservada para teste |
| `curva_aprendizado.png` | Desempenho em função do volume de treino disponível |
| `validacao_n_estimators.png` | Acurácia em função do número de árvores |
| `estabilidade_importancia_n_estimators.png` | Estabilidade do vetor de importância por número de árvores |
| `validacao_criterios_k.png` | Os quatro critérios de validação do número de agrupamentos |
| `ranking_combinado_k.png` | Ordenamento médio combinado dos quatro critérios |
| `cluster_jogadores.png` | Perfil técnico dos agrupamentos |
| `resultados_preliminares_jogadores.csv` | Atleta anonimizado, seleção, posição, agrupamento e indicadores |

---

## Reprodutibilidade

Todas as etapas com componente aleatório usam semente fixa (`random_state=42`), de modo que execuções sucessivas sobre a mesma base produzem os mesmos resultados. Pequenas diferenças podem aparecer entre versões distintas do scikit-learn, sobretudo na distribuição das partidas entre as dobras da validação cruzada.

---

## Uso de ferramentas de inteligência artificial

Durante o desenvolvimento deste trabalho, foram utilizadas ferramentas de inteligência artificial como apoio na revisão e depuração do código, no alinhamento dos procedimentos de validação, na sugestão de referências bibliográficas e na revisão de linguagem. Todas as referências foram verificadas nas fontes originais e os resultados foram reproduzidos em ambiente próprio. O autor assume total responsabilidade pelo conteúdo do trabalho. Este arquivo README foi redigido com apoio dessas ferramentas, a partir das descrições e informações fornecidas pelo autor, e posteriormente revisado por ele.

---

## Licença

Código disponibilizado sob a licença MIT. Os dados pertencem à Fédération Internationale de Volleyball e são aqui utilizados apenas para fins acadêmicos.
