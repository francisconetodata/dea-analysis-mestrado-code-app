import base64
import os
import time
from io import BytesIO

import matplotlib.pyplot as plt


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


def delete_html_output():
    os.remove('./templates/output.html')


def exists_output_html():
    var_aux = True
    while var_aux:
        if os.path.isfile('./templates/output.html'):
            var_aux = False
        else:
            time.sleep(1)
