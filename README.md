#Projeto de Classificação de Sentimentos - IMDb Reviews

Este projeto utiliza aprendizado de máquina e algumas técnicas de processamento de linguagem natural para classificar sentimentos em avaliações de filmes como positivos ou negativos. O objetivo é demonstrar a eficácia de técnicas de análise de sentimentos em dados textuais.

##Como foi feito?
- Pré-processamento de texto incluindo remoção de stopwords e stemming;
- Geração de embeddings semânticos com SenteceTransformer;
- Treinamento e avaliação de um modelo de Regressão Logística;
- Validação cruzada com visualização de curvas ROC;
- Interface interativa com Streamlit para classificação em tempo real;

##Linguagens e libs
- Python
- Bibliotecas: datasets, nltk, SentenceTransformer, scikit-learn, matplotlib, numpy, joblib, Streamlit

##Como executar o projeto
1. Clone este repositório
```
git clone https://github.com/PedroHenriquedp/machine_learning_project.git
cd machine_learning_project
```

2. Instale as dependências do projeto com
```
pip install -r requirements.txt
```

3. Execute o site localmente:
```
streamlit run app.py
```

##Link para o projeto no Google Colab
Você pode acessar o nosso laboratório virtual!! onde fizemos experimentações no Colab através deste link [Colab](https://colab.research.google.com/drive/1DQ4Xra11uCROCeMSsmAYNWF1unCVTNND?usp=sharing)

##Exemplo de Uso
No [site](https://am20243epu.streamlit.app/) desenvolvido, insira uma avaliação de filme no campo de texto e clique em classificar para processar e aguarde. O modelo retornará se o sentimento é positivo ou negativo, demonstrando a classificação de forma interativa!!

