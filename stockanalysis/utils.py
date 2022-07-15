import base64
import os
import time
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from pandas_datareader.data import DataReader


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_plot(x, y, ticker):
    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 4))
    plt.title(f'Série temporal do preço de fechamento da empresa: {ticker}')
    plt.xlabel('Datas')
    plt.ylabel('Preço de Fechamento ajustado')
    plt.tight_layout()
    plt.xticks(rotation=0)
    plt.plot(x, y)
    graph = get_graph()
    return graph


def create_output_html(ticker, data_inicial, data_final):
    import quantstats as qs
    ibov = DataReader(ticker, 'yahoo', data_inicial, data_final)
    qs.extend_pandas()
    serie = ibov['Adj Close'].diff()/ibov["Adj Close"].shift(1)
    eee = qs.reports.html(
        serie, '^BVSP', title=f' Análise de {ticker} em relação ao BVSP ')
    soup = BeautifulSoup(
        eee, "html.parser")
    with open("./templates/output.html", "w", encoding='utf-8') as file:
        # prettify the soup object and convert it into a string
        file.write(str(soup.prettify()))


def delete_html_output():
    os.remove('./templates/output.html')


def exists_output_html():
    var_aux = True
    while var_aux:
        if os.path.isfile('./templates/output.html'):
            var_aux = False
        else:
            time.sleep(1)
