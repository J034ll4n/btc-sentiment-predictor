📊 BTC Sentiment Predictor
Este projeto usa machine learning para prever a direção do preço do Bitcoin, combinando análise de sentimento de tweets e dados históricos de preços. A ideia é simples: se o sentimento nas redes sociais for majoritariamente positivo ou negativo, isso pode influenciar a movimentação do preço do BTC. Com base nessa lógica, o modelo aprende a prever se o próximo movimento será de alta ou baixa.

🚀 Tecnologias Utilizadas
Python

Pandas, NumPy

Scikit-learn

Joblib

Jupyter Notebook e VS Code

Fontes de dados: CoinGecko (preços do Bitcoin) + X (ex-Twitter) (sentimento de tweets)

🧠 Etapas do Projeto
1. Coleta de Dados
btc_coingecko.csv: Histórico de preços do Bitcoin, contendo dados de preço e timestamp.

tweets_btc_com_sentimento.csv: Tweets coletados com a análise de sentimento (positivo, negativo, neutro) usando técnicas de NLP.

2. Pré-processamento dos Dados
Conversão das datas para o formato datetime para garantir alinhamento correto dos dados.

Merge de dados: Utilização da função merge_asof para combinar os tweets com os preços históricos do Bitcoin.

Engenharia de Features:

Codificação dos sentimentos com LabelEncoder para transformar as categorias de sentimento em números.

Criação da variável target: Se o preço do Bitcoin no dia seguinte é superior ao preço atual, o movimento é classificado como alta (1); caso contrário, baixa (0).

3. Modelagem e Avaliação
Modelo: Utilização do RandomForestClassifier com 100 árvores (n_estimators=100).

Divisão dos dados: 80% para treino e 20% para teste.

Avaliação do modelo com métricas de precisão, recall e f1-score para entender o desempenho do modelo.

4. Exportação do Modelo
Modelo treinado salvo como: modelo_btc_random_forest.pkl para reutilização futura.

Scaler também é salvo, permitindo transformar dados novos de forma consistente com os dados de treinamento.
