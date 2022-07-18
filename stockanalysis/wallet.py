import base64
import os
import time
import warnings

import sqlite3 # Pacote do banco de dados
from datetime import datetime
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas_datareader.data import DataReader
from scipy import stats

from .pydea.dea import DEAProblem
from .models import Stock, PriceDataStocks

warnings.filterwarnings('ignore')


# For time stamps

# For reading stock data from yahoo


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png1 = buffer.getvalue()
    graph = base64.b64encode(image_png1)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_graph1():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png2 = buffer.getvalue()
    graph = base64.b64encode(image_png2)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_graph2():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png3 = buffer.getvalue()
    graph = base64.b64encode(image_png3)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_plot(x, y, ticker):
    plt.switch_backend('AGG')
    pass
    graph = get_graph()
    return graph


def wallet_visualizar(ticker, start, end):
    plt.switch_backend('AGG')
    for stock in ticker:
        globals()[stock] = DataReader(stock, 'yahoo', start, end)
    plt.figure(figsize=(25, 11))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    for i, company in enumerate(ticker, 1):
        plt.subplot(5, 3, i)
        globals()[company]['Adj Close'].plot(color='purple')
        plt.ylabel('Preço de \n Fechamento')
        plt.xlabel(None)
        plt.title(f"Preço de fechamento de: {ticker[i - 1]}")
    plt.tight_layout(pad=1.11)
    graph = get_graph()
    plt.switch_backend('AGG')
    for company in ticker:
        globals()[company]['Daily Return'] = globals()[
            company]['Adj Close'].pct_change()
    plt.figure(figsize=(25, 11))
    for i, company in enumerate(ticker, 1):
        plt.subplot(5, 3, i)
        globals()[company]['Daily Return'].dropna().plot(color='purple')
        plt.ylabel('Retorno diário')
        plt.title(f'Retorno diário: \n {ticker[i - 1]}')
        plt.xlabel("data")
    plt.tight_layout(pad=1.11)
    graph1 = get_graph1()
    plt.switch_backend('AGG')
    plt.figure(figsize=(25, 11))
    for i, company in enumerate(ticker, 1):
        plt.subplot(5, 3, i)
        sns.histplot(
            globals()[company]['Daily Return'].dropna(), bins=100, color='purple')
        plt.ylabel('Retorno diário')
        plt.title(f'Ticker: {ticker[i - 1]}')
        plt.xlabel(None)

    plt.tight_layout(pad=1.11)
    graph3 = get_graph2()
    plt.switch_backend('AGG')
    closing_df = DataReader(ticker, 'yahoo', start, end)['Adj Close']
    rets = closing_df.pct_change()
    rets_mult = rets+1
    rets_mult = rets_mult.dropna()
    simula_cada = rets_mult.cumprod()
    plt.figure(figsize=(25, 11))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(ticker, 1):
        plt.subplot(5, 3, i)
        simula_cada[company].plot(color='purple')
        plt.ylabel('Valor \n em Reais')
        plt.xlabel(None)
        plt.title(f"Simulação de investimento \n unitário em: {ticker[i - 1]}")

    plt.tight_layout(pad=1.11)
    graph4 = get_graph3()
    return [graph, graph1, graph3, graph4]


def get_graph3():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png4 = buffer.getvalue()
    graph = base64.b64encode(image_png4)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_graph4():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png5 = buffer.getvalue()
    graph = base64.b64encode(image_png5)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def mark_carteira(ticker, 
                  data_inicio, 
                  data_final, 
                  risk_free, 
                  metodo_dea,
                  data_simulacao):
    livre_de_risco = risk_free  # Basicamente a taxa básica de juros
    names_tickers_aux = ticker
    conn = sqlite3.connect('db.sqlite3')
    #cursor = conn.cursor()

    dados = pd.read_sql(f"""
                        SELECT ss.date_ref, ss.price_close, a.symbol_stock  
                        FROM stockanalysis_pricedatastocks ss
                        inner join stockanalysis_stock a on a.id = ss.stock_id
                        where ss.date_ref > '{data_inicio}' AND 
                        ss.date_ref <= '{data_final}'
                        AND a.symbol_stock in {tuple(ticker)}
                        """,conn)

    dados.index = pd.to_datetime( dados['date_ref'])
    dados = dados.drop(columns=['date_ref'])
    df = dados[dados['symbol_stock']==ticker[0]]
    df = df.drop(columns='symbol_stock')

    for i in ticker:
        if i == ticker[0]:
            continue
        else:
            df = df.merge(
                dados[dados['symbol_stock']==i]['price_close'],
                how='inner',
                left_index=True,
                right_index=True
            )
    df.columns = ticker
    dados = df.copy()
    print(df)
    #f (len(dados.dropna())/len(dados)) > (per_01/100):
    #    coluns = dados.isnull().sum()>=((per_01/100))*len(dados)
    #    empresas_retiradas = list(coluns[coluns].index.values)
    #    empresas_retiradas_str = " , ".join(empresas_retiradas)
    #    dados = dados.drop(columns=(coluns[coluns].index.values))
    #    ticker = list(dados.columns)
    #if (len(dados.pct_change()[dados.pct_change()==0])/len(dados.pct_change())) > (per_02/100):
    #    coluns = dados.pct_change()[dados.pct_change()==0].sum()>=((per_02/100))*len(dados.pct_change())
    #    empresas_retiradas_ = list(coluns[coluns].index.values)
    #    empresas_retiradas_str_ = " , ".join(empresas_retiradas_)
    #    dados = dados.drop(columns=(coluns[coluns].index.values))
    #   ticker = list(dados.columns)
    #print(empresas_retiradas_str_)
    #ticker = names_tickers_aux
    df_ibov_data = pd.DataFrame()
    data_final = dados.index.max()
    data_inicio = dados.index.min()
    df_ibov_data = pd.read_sql(f"""
                        SELECT ss.date_ref, ss.price_close
                        FROM stockanalysis_pricedatastocks ss
                        inner join stockanalysis_stock a on a.id = ss.stock_id
                        where ss.date_ref > '{data_inicio}' AND 
                        ss.date_ref <= '{data_final}'
                        AND a.symbol_stock = '^BVSP'
                        """,conn)
    df_ibov_data.index = pd.to_datetime(df_ibov_data['date_ref'])
    df_ibov_data.columns = ['date_ref','^BVSP']
    df_ibov_data = df_ibov_data.drop(columns='date_ref')
    df_ibov_data = df_ibov_data.dropna()
    df_ibov_data_ = (df_ibov_data.astype(float)/df_ibov_data.iloc[0])
    df_aux_ibov = df_ibov_data.astype(float).pct_change()
    df_aux_ibov = df_aux_ibov.dropna()
    dados___ = dados.copy()
    dados___.sort_index(inplace=True)
    retornos = dados___.pct_change()
    retornos_medios = retornos.mean()
    cov_matrix = retornos.cov()
    corrl_p = retornos.dropna().corr(method='pearson').round(3).reset_index()
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(corrl_p.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[corrl_p[f'{i}'] for i in list(corrl_p.columns)],
               fill_color='lavender',
               align='left'))
    ])
    fig.update_layout(
    autosize=True,
    paper_bgcolor='rgb(255, 255, 255)',
    )
    corrl_p = fig.to_html()
    corrl_s = retornos.dropna().corr(method='spearman').round(3).reset_index()
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(corrl_s.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[corrl_s[f'{i}'] for i in list(corrl_s.columns)],
               fill_color='lavender',
               align='left'))
    ])
    fig.update_layout(
    autosize=True,
    paper_bgcolor='rgb(255, 255, 255)',
    )
    corrl_s = fig.to_html()
    corrl_k = retornos.dropna().corr(method='kendall').round(3).reset_index()
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(corrl_k.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[corrl_k[f'{i}'] for i in list(corrl_k.columns)],
               fill_color='lavender',
               align='left'))
    ])
    fig.update_layout(
    autosize=True,
    paper_bgcolor='rgb(255, 255, 255)',
    )
    corrl_k = fig.to_html()
    retornos_prod = retornos.dropna() + 1
    tamanho = len(retornos_prod)
    periodo_01 = retornos_prod.iloc[0:int(tamanho)].product(axis=0)
    periodo_02 = retornos_prod.iloc[2*int(
        tamanho/3):int(tamanho)].product(axis=0)
    periodo_03 = retornos_prod.iloc[int(
        (tamanho/3)):int(tamanho)].product(axis=0)
    volatilidade = retornos.dropna().std()
    print(retornos)
    retornos__ = retornos.merge(df_aux_ibov, how='inner', left_index=True,
                                right_index=True)
    print(retornos__)
    retornos__.columns = ticker + ['BVSP']
    retornos__ = retornos__.astype(float).dropna()
    betas = []
    for i in ticker:
        try:
            values_x = retornos__[f'{i}'].dropna().values
            values_y = retornos__['BVSP'].dropna().values
            LR = stats.linregress(
                values_x,
                values_y)
        except ValueError:
            betas.append(1.000)
        else:
            LR = stats.linregress(
                retornos__[f'{i}'].dropna().values,
                retornos__['BVSP'].dropna().values)
            betas.append(round(LR[0], 4))
    betas_series = pd.Series(betas, index=ticker)
    dfk = pd.concat([periodo_01, periodo_02, periodo_03,
                    volatilidade, betas_series], axis=1).round(3)
    dfk = dfk.reset_index()
    dfk.columns = ['Ticker', 'Retornos 1', 'Retornos 2',
                   'Retornos 3', 'Volatilidade', 'Beta']
    primDEA = DEAProblem(dfk[['Volatilidade', 'Beta']], dfk[['Retornos 1', 'Retornos 2',
                                                             'Retornos 3']], returns=metodo_dea)
    #dfk.to_excel('exemplodea.xlsx')
    primResults = primDEA.solve()
    dfk['Eficiência'] = primResults['Efficiency'].round(3)
    print(dfk)
    dados_dea_x = dfk.copy()
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(dfk.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[dfk[f'{i}'] for i in list(dfk.columns)],
               fill_color='lavender',
               align='left'))
    ])
    fig.update_layout(
    
    paper_bgcolor='rgb(255, 255, 255)'
    )
    dfkk = fig.to_html()
    dfk.to_excel('dea.xlsx')
    names_tickers_aux = (list(dfk[dfk['Eficiência'] >= 0.999999]['Ticker'].values))
    ticker = names_tickers_aux
    dados = dados[ticker]
    num_portfolios = 4500
    resultados = np.zeros((len(names_tickers_aux)+3, num_portfolios))
    dados.sort_index(inplace=True)
    retornos = dados.pct_change()
    retornos_medios = retornos.mean()
    preços = pd.DataFrame()

    for t in names_tickers_aux:
        
        dados_t = pd.read_sql(f"""
                        SELECT ss.date_ref, ss.price_close  
                        FROM stockanalysis_pricedatastocks ss
                        inner join stockanalysis_stock a on a.id = ss.stock_id
                        where ss.date_ref > '{data_inicio}' AND 
                        ss.date_ref <= '{data_final}'
                        AND a.symbol_stock = '{t}'
                        """,conn)
        dados_t.columns = ['data', t ]
        dados_t.index = pd.to_datetime(dados_t['data'])
        dados_t = dados_t.drop(columns=['data'])
        preços[t] = dados_t
    print(preços)
    preços = preços.dropna()
    preços_normalizados = (preços/preços.iloc[0])
    cov_matrix = retornos.cov()
    for i in range(num_portfolios):
        if i == 4:
            pesos = np.zeros(len(names_tickers_aux)) + 1
            pesos /= np.sum(pesos)
            retorno_portfolio = np.sum(retornos_medios * pesos)*252
            desvio_padrao_portfolio = np.sqrt(
                np.dot(pesos.T, np.dot(cov_matrix, pesos)))*np.sqrt(252)

            resultados[0, i] = retorno_portfolio
            resultados[1, i] = desvio_padrao_portfolio
            resultados[2, i] = (resultados[0, i] / resultados[1, i]
                                ) - livre_de_risco  # Índice de Sharpe
            for j in range(len(pesos)):
                resultados[j+3, i] = pesos[j]
        else:

            pesos = np.array(np.random.random(len(names_tickers_aux)))
            pesos /= np.sum(pesos)
            retorno_portfolio = np.sum(retornos_medios * pesos)*252
            desvio_padrao_portfolio = np.sqrt(
                np.dot(pesos.T, np.dot(cov_matrix, pesos)))*np.sqrt(252)

            resultados[0, i] = retorno_portfolio
            resultados[1, i] = desvio_padrao_portfolio
            resultados[2, i] = (resultados[0, i] / resultados[1, i]
                                ) - livre_de_risco  # Índice de Sharpe
            for j in range(len(pesos)):
                resultados[j+3, i] = pesos[j]
    resultados_frame = pd.DataFrame(
        resultados.T, columns=(['retorno', 'desvio_padrao', 'sharpe']+ticker))
    #retorno = resultados_frame.iloc[resultados_frame['retorno'].idxmax()]
    max_sharpe_port = resultados_frame.iloc[resultados_frame['sharpe'].idxmax(
    )]
    min_vol_port = resultados_frame.iloc[resultados_frame['desvio_padrao'].idxmin(
    )]
    unif_cart = resultados_frame.iloc[4]
    sharpe_unif = round(unif_cart['sharpe'], 4)
    mark_sharpe = round(max_sharpe_port['sharpe'], 4)
    #print(sharpe_unif, mark_sharpe)
    dados = dados.dropna()
    retorno_carteira_unif = 0

    for i in range(len(names_tickers_aux)):
        retorno_carteira_unif += (unif_cart[names_tickers_aux[i]]
                                  * preços_normalizados[names_tickers_aux[i]])
    retorno_carteira = 0

    for i in range(len(names_tickers_aux)):
        retorno_carteira += (max_sharpe_port[names_tickers_aux[i]]
                             * preços_normalizados[names_tickers_aux[i]])

    plt.switch_backend('AGG')
    plt.figure(figsize=(18, 9))
    plt.scatter(resultados_frame.desvio_padrao, resultados_frame.retorno,
                c=resultados_frame.sharpe,
                marker='o',
                facecolors='none', edgecolors='black')
    plt.xlabel('Volatilidade')
    plt.ylabel('Retorno')
    plt.colorbar()
    plt.tight_layout(pad=1.11)
    plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(
        5, 1, 0), label='Máx Sharpe', color='blue', s=850)
    plt.scatter(min_vol_port[1], min_vol_port[0], marker=(
        5, 1, 0), label='Mínima Variância', color='darkorange', s=850)
    plt.scatter(unif_cart[1], unif_cart[0], marker=(
        5, 1, 0), label='Uniforme 1/N', color='red', s=850)
    plt.legend(fontsize='medium')
    graph = get_graph5()
    fffff = pd.DataFrame(max_sharpe_port[3:(len(names_tickers_aux)+3)])
    fffff = fffff.reset_index()
    fffff.columns = ['Ticker', 'Dados']
    #plt.switch_backend('AGG')
    #preços_normalizados.plot(figsize=(18, 9))
    #plt.ylabel('Preços Normalizados')
    #plt.tight_layout(pad=1.11)
    #plt.xlabel('Data')
    fig_nr = px.line(
        preços_normalizados,
        title='Preços Normalizados das ações'
    )
    graph1 = fig_nr.to_html()#get_graph6()
    #plt.switch_backend('AGG')
    #plt.figure(figsize=(18, 9))
    #fffff.plot(x='Ticker', y='Dados', kind='bar')
    #plt.xticks(rotation=90)
    #plt.xlabel('Ativo')
    #plt.ylabel('Pesos')
    #plt.tight_layout(pad=1.45, h_pad=1.46, w_pad=1.56)
    fffff['Dados'] = fffff['Dados'].round(3)
    barras_pl = px.bar(
        fffff,
        x = 'Ticker',
        y = 'Dados',
        title= 'Pesos do portfólio que maximiza o índice sharpe.',
        text_auto= True
    )
    graph2 = barras_pl.to_html() #get_graph7()
    df_final_merge = pd.merge(df_ibov_data_,retorno_carteira,left_index=True,right_index=True, how='inner')
    df_final_merge_ = df_final_merge.merge(retorno_carteira_unif, left_index=True,right_index=True, how='inner')
    df_final_merge_.columns = ['Ibovespa','DEA(M)','DEA(1/N)']
    df_final_merge_ = df_final_merge_.dropna().astype(float)
    #plt.switch_backend('AGG')
    #plt.figure(figsize=(18, 9))
    #df_ibov_data_.plot(label='BVSP', legend='better')
    #retorno_carteira.plot(label='DEA(M)', legend='better')
    #retorno_carteira_unif.plot(label='DEA(1/N)', legend='better')
    #plt.xlabel('Data')
    #plt.ylabel('Retorno')
    #plt.tight_layout(pad=1.20, h_pad=1.35, w_pad=1.35)
    fig_simulador = px.line(
        df_final_merge_,
        #x= df_final_merge_.index,
        #y= ['Ibovespa','DEA(M)','DEA(1/N)'],
        title='Comportamento das carteiras durante o período de análise.'
    )
    graph3 = fig_simulador.to_html()# get_graph8()

    dados_corr = pd.merge(retorno_carteira, df_ibov_data_, how='inner',
                          left_index=True, right_index=True)
    dados_corr.columns = ['Carteira', 'Ibov']
    dados_corr = dados_corr.dropna()

    LR = stats.linregress(
        dados_corr['Carteira'].astype(float).values, dados_corr['Ibov'].astype(float).values)
    plt.switch_backend('AGG')
    plt.figure(figsize=(18, 9))
    sns.regplot(dados_corr['Carteira'].astype(float).values, dados_corr['Ibov'].astype(float).values)
    plt.xlabel("Retorno do BVSP")
    plt.ylabel("Retorno da Carteira")
    plt.title("Retorno da DEA(M) vs Retorno do BVSP - Regressão Linear")
    (beta, alfa) = LR[0:2]
    graph4 = get_graph9()
    dados_corr = pd.merge(retorno_carteira_unif, df_ibov_data_, how='inner',
                          left_index=True, right_index=True)
    dados_corr.columns = ['Carteira', 'Ibov']
    dados_corr = dados_corr.dropna()

    LR = stats.linregress(
        dados_corr['Carteira'].astype(float).values, dados_corr['Ibov'].astype(float).values)
    plt.switch_backend('AGG')
    plt.figure(figsize=(18, 9))
    sns.regplot(dados_corr['Carteira'].astype(float).values, dados_corr['Ibov'].astype(float).values)
    plt.xlabel("Retorno do BVSP")
    plt.ylabel("Retorno da Carteira - DEA(1/N)")
    plt.title("Retorno da DEA(1/N) vs Retorno do BVSP - Regressão Linear")
    (beta1, alfa1) = LR[0:2]
    graph5 = get_graph10()

    fffff.columns = ['Empresa', 'Peso Portifólio']
    fffff__= fffff.copy()
    fffff__.columns = ['Empresa', 'Peso']
    fffff__['Peso'] = fffff__['Peso'].round(3)
    nomes = fffff[['Empresa']].copy()
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(nomes.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[nomes[f'{i}'] for i in list(nomes.columns)],
               fill_color='lavender',
               align='center'))
    ])
    fig.update_layout(
    autosize=True,
    paper_bgcolor='rgb(255, 255, 255)',
    )
    nomess = fig.to_html()
    fffff_ = fffff
    #print(nomes)
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(fffff_.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[fffff_[f'{i}'] for i in list(fffff_.columns)],
               fill_color='lavender',
               align='left'))
    ])
    fig.update_layout(
    autosize=True,
    paper_bgcolor='rgb(255, 255, 255)',
    )
    fffff = fig.to_html()
    tickers_ibov_ = list(fffff__.Empresa)
    #print(tickers_ibov_)
    #print(list(fffff__.Peso))
    dados_yahoo = pd.read_sql(f"""
                        SELECT ss.date_ref, ss.price_close, a.symbol_stock  
                        FROM stockanalysis_pricedatastocks ss
                        inner join stockanalysis_stock a on a.id = ss.stock_id
                        where ss.date_ref > '{data_final}' AND 
                        ss.date_ref <= '{data_simulacao}'
                        AND a.symbol_stock in {tuple(tickers_ibov_)}
                        """,conn)

    dados_yahoo.index = pd.to_datetime(dados_yahoo['date_ref'])
    dados_yahoo = dados_yahoo.drop(columns=['date_ref'])
    df = dados_yahoo[dados_yahoo['symbol_stock']==ticker[0]]
    df = df.drop(columns='symbol_stock')

    for i in ticker:
        if i == ticker[0]:
            continue
        else:
            df = df.merge(
                dados_yahoo[dados_yahoo['symbol_stock']==i]['price_close'],
                how='inner',
                left_index=True,
                right_index=True
            )
    df.columns = tickers_ibov_
    dados_yahoo = df
    
    ibov = pd.read_sql(f"""
                        SELECT ss.date_ref, ss.price_close
                        FROM stockanalysis_pricedatastocks ss
                        inner join stockanalysis_stock a on a.id = ss.stock_id
                        where ss.date_ref > '{data_final}' AND 
                        ss.date_ref <= '{data_simulacao}'
                        AND a.symbol_stock = '^BVSP'
                        """,conn)
    ibov.index = pd.to_datetime(ibov['date_ref'])
    ibov.columns = ['date_ref','^BVSP']
    ibov = ibov.drop(columns='date_ref')
    ibov = ibov / ibov.iloc[0]
    #print(dados_yahoo)
    #print(dados_yahoo.columns)
    #dados_yahoo.dropna(how='all', inplace=True)
    #dados_yahoo.dropna(axis=1, inplace=True, thresh=246)
    retorno = dados_yahoo.pct_change()
    retorno_acumulado = (1 + retorno).cumprod()
    retorno_acumulado.iloc[0] = 1
    #print(retorno)
    #print(retorno_acumulado)
    carteiram = list(fffff__.Peso)* retorno_acumulado.loc[: , list(dados_yahoo.columns)]
    carteiram['saldo'] = carteiram.sum(axis=1)
    carteiram["retorno"] = carteiram['saldo'].pct_change()
    carteiran = (1/len(nomes) )* retorno_acumulado.loc[: ,  list(dados_yahoo.columns)]
    carteiran['saldo'] = carteiran.sum(axis=1)
    carteiran["retorno"] = carteiran['saldo'].pct_change()
    df_final_merge = pd.merge(ibov,carteiram['saldo'],left_index=True,right_index=True, how='inner')
    df_final_merge_ = df_final_merge.merge(carteiran["saldo"], left_index=True,right_index=True, how='inner')
    df_final_merge_.columns = ['Ibovespa','DEA(M)','DEA(1/N)']
    #plt.switch_backend('AGG')
    #plt.figure(figsize=(18, 9))
    #plt.tight_layout(pad=1.20, h_pad=1.35, w_pad=1.35)
    #(ibov*1).plot(label='Ibov', legend='better')
    #carteiram['saldo'].plot(label='DEA(M)', legend='better')
    #carteiran['saldo'].plot(label='DEA(1/N)', legend='better')
    #plt.xlabel("Data")
    #plt.ylabel("Valor resultante")
    #plt.title("Simulação de investimento unitário em cada carteira - Comparação")
    fig_simulador = px.line(
        df_final_merge_,
        x=df_final_merge_.index,
        y= ['Ibovespa','DEA(M)','DEA(1/N)'],
        title='Simulação de investimento unitário em cada carteira - Comparação'
    )
    graph6 = fig_simulador.to_html()
    return [graph, fffff, graph1, graph2, graph3, graph4, beta, alfa,
            corrl_p, corrl_s, corrl_k, dfkk, nomess, graph5, beta1, alfa1,
            sharpe_unif, mark_sharpe,graph6, data_inicio, data_final, data_simulacao, 
            [], [],dados_dea_x]


def get_graph5():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png6 = buffer.getvalue()
    graph = base64.b64encode(image_png6)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_graph6():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png7 = buffer.getvalue()
    graph = base64.b64encode(image_png7)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_graph7():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png8 = buffer.getvalue()
    graph = base64.b64encode(image_png8)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_graph8():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png9 = buffer.getvalue()
    graph = base64.b64encode(image_png9)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_graph9():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png10 = buffer.getvalue()
    graph = base64.b64encode(image_png10)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_graph10():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png11 = buffer.getvalue()
    graph = base64.b64encode(image_png11)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph
def get_graph11():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png12 = buffer.getvalue()
    graph = base64.b64encode(image_png12)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph