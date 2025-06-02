# 📊 BTC Sentiment Predictor

Este projeto utiliza **machine learning** para prever a direção do preço do Bitcoin com base na **análise de sentimento de tweets** e no **histórico de preços do BTC**.

A ideia é simples: se o sentimento geral nas redes sociais está positivo ou negativo, isso pode influenciar o movimento de preço do BTC. Com essa lógica, o modelo aprende a classificar se o próximo movimento será de **alta** ou **baixa**.

---

## 🚀 Tecnologias Utilizadas

- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook e VS Code
- Dados de: CoinGecko + Tweets

---

## 🧠 Etapas do Projeto

### 1. Coleta de Dados

- **btc_coingecko.csv**: Histórico de preços do Bitcoin (data, preço).
- **tweets_btc_com_sentimento.csv**: Tweets com análise de sentimento extraída via NLP (`positivo`, `negativo`, etc).

### 2. Pré-processamento

- Conversão de datas para `datetime`.
- Merge de dados via `merge_asof` para alinhar tweets com os horários dos preços.
- Engenharia de features:
  - Conversão do sentimento em números (`LabelEncoder`).
  - Criação da variável `target`: se o preço do dia seguinte é maior que o atual → 1 (alta), senão → 0 (baixa).

### 3. Modelagem

- Modelo: `RandomForestClassifier` com `n_estimators=100`.
- Divisão dos dados: 80% treino, 20% teste.
- Treinamento e avaliação.

### 4. Exportação

- Modelo salvo como: `modelo_btc_random_forest.pkl`.
- Scaler salvo para futura transformação de dados reais.

---

## 📈 Resultados

### Métricas do modelo (exemplo real):

```text
              precision    recall  f1-score   support

           0       0.81      0.84      0.82        19
           1       0.87      0.84      0.85        25

    accuracy                           0.84        44
   macro avg       0.84      0.84      0.84        44
weighted avg       0.84      0.84      0.84        44
