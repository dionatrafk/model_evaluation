# Informações

As pastas:
-  **http_requests_nasa** 
- **http_requests_wikipedia**

Contém os datasets da NASA e Wikipédia correspondentes aos intervalos de previsão. Por exemplo, trace60.csv contém os dados necessários para realizar previsões para os próximos 60 minutos.

Os dados originais do dataset Wikipédia: http://www.wikibench.eu/wiki/2007-09/

- **algorithms**

Contém os códigos fontes dos modelos de Aprendizado de Máquina (MLP, GRU, ARIMA) e seus otimizadores de hiperparâmetros (*Grid Search* (**grid_search.py**), *Tree of parzen estimators (TPE)* (**hypeas_gru.py** e **hypeas_mlp.py**) para execução local.

Para a execução dos algoritmos serão necessários um ambiente com as seguintes bibliotécas abaixo:

- **Keras** https://keras.io/
- **SKlearn** https://scikit-learn.org
- **Hyperopt** http://hyperopt.github.io/hyperopt/
- **Statsmodels** https://www.statsmodels.org/stable/index.html
- **Matplotlib** https://matplotlib.org/
- **Pandas** https://pandas.pydata.org/
- **Numpy** http://www.numpy.org/

Os algoritmos abaixo, estão disponíveis como alternativa para  execução on-line:
- **MLP_exec.ipynb**
- **GRU_exec.ipynb**
- **ARIMA_exec.ipynb**

Correspondem ao código fonte de cada modelo de Aprendizado de Máquina. Estão no formato semelhante ao Jupyter notebook. 
Ao acessar a página do arquivo, selecione *Open in Colab*, para acessar o modelo em modo de execução no ambiente do *Google Colaboratory*.
