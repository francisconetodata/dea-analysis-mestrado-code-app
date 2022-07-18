from concurrent.futures import thread
import json
import os
import time
import io
from threading import Thread
#from somewhere import hand
import django_excel as excel
import openpyxl
from django.contrib import messages


import pandas as pd
from django.http import HttpResponse, HttpResponseRedirect, HttpRequest
from django.template.response import TemplateResponse

from django.shortcuts import render
from pandas_datareader.data import DataReader
from .form import DownloadP, StockInput, StockInputCart
from django import forms
from .bot import alguemfezanalise , alguemfezlogin, alguemfezanalisedea, alguemfezdownload

from .utils import create_output_html, delete_html_output, get_plot
from .wallet import mark_carteira
from .models import Stock, PriceDataStocks
from .atualizar_db import atualizar_db
from .pydea.dea import DEAProblem

class UploadFileForm(forms.Form):
    dea_method = forms.ChoiceField(label='Método DEA',
                                   choices=(('VRS', 'VRS'), ('CRS', 'CRS')))
    file = forms.FileField(allow_empty_file=False)
    
 
def download_view(request):
    form = StockInput(request.POST or None)
    if str(request.method) == 'POST':
        if form.is_valid():
            while (os.path.isfile('./templates/output.html') == True):
                delete_html_output()
                time.sleep(3)
            data_inicial = form.cleaned_data['data_inicial']
            data_final = form.cleaned_data['data_final']
            ticker = form.cleaned_data['ticker']
            # Setando o período
            end = str(data_final)
            start = str(data_inicial)
            ticker = str(ticker)
            create_output_html(ticker, start, end)
            time.sleep(15)
            ibov = DataReader(ticker, 'yahoo', start, end)
            ibov = ibov.iloc[:, 1:2]
            ibov.columns = ['ValorFechamento']
            yy = ibov['ValorFechamento'].values
            ibov = ibov.reset_index()
            ibov['Date'] = pd.to_datetime(ibov['Date'].astype(str))
            xx = ibov['Date'].values
            while (os.path.isfile('./templates/output.html') == False):
                time.sleep(3)
            chart = get_plot(xx, yy, ticker)
            form = StockInput()
        else:
            time.sleep(3)
            chart = []
            form = StockInput()
    else:
        time.sleep(3)
        chart = []
        form = StockInput()
    var_aux = True
    while var_aux:
        if os.path.isfile('./templates/output.html'):
            var_aux = False
            return render(request, 'download.html', {'form': form,
                                                     'chart': chart})
        else:
            time.sleep(2)



def relatorio_carteira(request):
    form = StockInputCart(request.POST or None)
    if str(request.method) == 'POST':
        if form.is_valid():
            data_inicial = form.cleaned_data['data_inicial']
            data_final = form.cleaned_data['data_final']
            data_simulacao = form.cleaned_data['data_simulacao']
            ticker = form.cleaned_data['ticker']
            risk_free = form.cleaned_data['risk_free']
            metodo_dea = form.cleaned_data['dea_method']
            
            selecionar_todas = False #form.cleaned_data['selecionar_todas']
            if selecionar_todas==True:
                ticker_query =list( Stock.objects.values_list('symbol_stock').order_by('symbol_stock'))
                ticker = []
                for i in range(len(ticker_query)):
                    ticker.append(Stock.objects.values_list('symbol_stock').order_by('symbol_stock')[i][0])
                    print(Stock.objects.values_list('symbol_stock').order_by('symbol_stock')[i][0])
                print(ticker)
            else:
                ticker = form.cleaned_data['ticker']
            risk_free = float(risk_free)
            end = str(data_final)
            start = str(data_inicial)
            chart = []
            chart2 = []
            chart3 = []
            chart4 = []
            mark = mark_carteira(ticker, 
                                 start, 
                                 end, 
                                 risk_free, 
                                 metodo_dea,
                                 data_simulacao)
            chart5 = mark[0]
            alguemfezanalise()
            portfolio_max = mark[1]
            chart6 = mark[2]
            chart7 = mark[3]
            chart8 = mark[4]
            chart9 = mark[5]
            beta = round(mark[6], 4)
            alfa = round(mark[7], 4)
            pearson = mark[8]
            spearman = mark[9]
            kendall = mark[10]
            dea = mark[11]
            empresas = mark[12]
            chart10 = mark[13]
            alfa1 = round(mark[14], 4)
            beta1 = round(mark[15], 4)
            sharp_unif = mark[16]
            sharp_m = mark[17]
            chart11 = mark[18]
            inicio = str(mark[19])[0:10]
            fim = str(mark[20])[0:10]
            simulador = str(mark[21])[0:10]
            empresas_ret = mark[22]
            empresas_ret_ = mark[23]
            dados_dea = mark[24]
            resultado_simulador = mark[25]
            mark = []
            import random
            numberee = random.sample(range(114444), k=1)[0]
            dados_dea.to_excel('var/tmp/django_cache/'+f'report_dea_{numberee}.xlsx')
            request.session['listing_list'] = [f'report_dea_{numberee}.xlsx']
            resultado_simulador = []
            mark = []
            form = StockInputCart()
            response = render(request, 'relatoriocarteira.html', {'form': form,
                                                      'chart': chart,
                                                      'chart2': chart2,
                                                      'chart3': chart3,
                                                      'chart4': chart4,
                                                      'chart5': chart5,
                                                      'chart6': chart6,
                                                      'chart7': chart7,
                                                      'chart8': chart8,
                                                      'chart9': chart9,
                                                      'chart10': chart10,
                                                      'alfa': alfa,
                                                      'beta': beta,
                                                      'portmark': portfolio_max,
                                                      'pearson': pearson,
                                                      'kendall': kendall,
                                                      'speaman': spearman,
                                                      'dea': dea,
                                                      'empresas': empresas,
                                                      'chart11': chart11,
                                                      'beta1': beta1,
                                                      'alfa1': alfa1,
                                                      'dea_method': metodo_dea,
                                                      'sharpe_unif': sharp_unif,
                                                      'inicio': inicio,
                                                      'fim': fim,
                                                      'empresas_ret_':empresas_ret_,
                                                      'simulador': simulador,
                                                      'empresas_ret': empresas_ret,                                                      
                                                      'sharpe_m': sharp_m,
                                                      'resultado_simulador':resultado_simulador})
            
            return response
        else:
            empresas = []
            sharp_unif = []
            sharp_m = []
            inicio = []
            fim = []
            dados_dea = []
            simulador = []
            empresas_ret = []
            empresas_ret_ = []
            chart = []
            chart2 = []
            chart3 = []
            metodo_dea = []
            chart4 = []
            chart5 = []
            chart6 = []
            chart8 = []
            chart7 = []
            chart9 = []
            alfa = []
            portfolio_max = []
            beta = []
            pearson = []
            spearman = []
            kendall = []
            dea = []
            chart10 = []
            alfa1 = []
            beta1 = []
            chart11 = []
            perc_min_02 = []
            resultado_simulador = []
            perc_min_01 = []
            form = StockInputCart()
    else:
        chart = []
        dea = []
        perc_min_02 = []
        perc_min_01 = []
        chart10 = []
        alfa1 = []
        beta1 = []
        chart2 = []
        empresas = []
        chart3 = []
        inicio = []
        dados_dea = []
        fim = []
        empresas_ret_ = []
        simulador = []
        resultado_simulador = []
        empresas_ret = []
        metodo_dea = []
        chart4 = []
        chart5 = []
        chart6 = []
        chart8 = []
        chart7 = []
        chart9 = []
        alfa = []
        pearson = []
        spearman = []
        kendall = []
        portfolio_max = []
        beta = []
        sharp_unif = []
        sharp_m = []
        chart11 = []
        form = StockInputCart()
    form = StockInputCart()
    response = render(request, 'relatoriocarteira.html', {'form': form,
                                                      'chart': chart,
                                                      'chart2': chart2,
                                                      'chart3': chart3,
                                                      'chart4': chart4,
                                                      'chart5': chart5,
                                                      'chart6': chart6,
                                                      'chart7': chart7,
                                                      'chart8': chart8,
                                                      'chart9': chart9,
                                                      'chart10': chart10,
                                                      'alfa': alfa,
                                                      'beta': beta,
                                                      'portmark': portfolio_max,
                                                      'pearson': pearson,
                                                      'kendall': kendall,
                                                      'speaman': spearman,
                                                      'dea': dea,
                                                      'empresas': empresas,
                                                      'chart11': chart11,
                                                      'beta1': beta1,
                                                      'alfa1': alfa1,
                                                      'dea_method': metodo_dea,
                                                      'sharpe_unif': sharp_unif,
                                                      'inicio': inicio,
                                                      'fim': fim,
                                                      'empresas_ret_':empresas_ret_,
                                                      'simulador': simulador,
                                                      'empresas_ret': empresas_ret,                                                      
                                                      'sharpe_m': sharp_m,
                                                      'dados_dea':dados_dea,
                                                      'resultado_simulador':resultado_simulador})
    return response

def download_relatorio_dea(request):    
    try:
        listing_list = request.session.get('listing_list')
        df_exx = pd.read_excel('var/tmp/django_cache/'+listing_list[0])
        response = HttpResponse(
                    content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename="{listing_list[0]}"'
        df_exx.to_excel(response)
    #os.remove(listing_list[0])
    except:
        messages.error('O arquivo não está mais disponível, favor refazer a análise.')
    else: 
        listing_list = request.session.get('listing_list')
        df_exx = pd.read_excel('var/tmp/django_cache/'+listing_list[0])
        response = HttpResponse(
                    content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename="{listing_list[0]}"'
        df_exx.to_excel(response)
        return response

def pacotes(request):
    pacotes = pd.read_excel('requirements.xlsx')
    pacotes.columns = ['Pacotes:']
    pacotes = pacotes.to_html()
    return render(request, 'sobre.html', {'pack': pacotes})

def app_atualizar(request):
    def task_2():
        atualizar_db()
    #t2 = Thread(target=task_2)
    #t2.start()
    alguemfezlogin()
    return render(request, 'app.html')

def download_view_p(request):
    form = DownloadP(request.POST or None)
    if str(request.method) == 'POST':
        if form.is_valid():
            alguemfezdownload()
            data_inicial = form.cleaned_data['data_inicial']
            data_final = form.cleaned_data['data_final']
            ticker = form.cleaned_data['ticker']
            end = str(data_final)
            start = str(data_inicial)
            dados = DataReader(ticker, 'yahoo', start=start, end=end)
            response = HttpResponse(
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = 'attachment; filename="dados.xlsx"'
            dados.to_excel(response)
            return response
        else:
            form = DownloadP()
    form = DownloadP()
    return render(request, 'downloadp.html', {'form': form})

def exemplo_dea(request):
    dados = pd.read_excel('exemplodea.xlsx')
    response = HttpResponse(
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="exemplodea.xlsx"'
    dados.to_excel(response,index=False)
    return response
    

def info_base(request):
    import sqlite3 # Pacote do banco de dados
    conn = sqlite3.connect('db.sqlite3')
    dados = pd.read_sql("""
                        SELECT  ss.name_stock ,
                        ss.symbol_stock ,
                        MAX((sp.date_ref )) as MaxDate,
                        MIN((sp.date_ref )) as MinDate,
                        COUNT(DISTINCT(sp.id)) AS QtdDadosDisponivel 
                        FROM stockanalysis_pricedatastocks as sp
                        INNER JOIN stockanalysis_stock ss on ss.id = sp.stock_id 
                        WHERE ss.symbol_stock NOT LIKE '^BVSP'
                        GROUP BY ss.name_stock , ss.symbol_stock
                        ORDER BY COUNT(DISTINCT(sp.id)) DESC,ss.name_stock 
                        """,conn)
    dados.columns = [
        'Nome Empresa',
        'Símbolo',
        'Data máxima disponível',
        'Data mínima disponível',
        'Qtd Dados Disponíveis'
    ]
    response = HttpResponse(
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="dados_base_resumo.xlsx"'
    dados.to_excel(response,index=False)
    return response


def upload_file(request):
    #form = UploadFileForm(request.POST or None or request.FILES)
    if str(request.method) == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            alguemfezanalisedea()
            excel_file = request.FILES["file"]
            dea_method = str(form.cleaned_data['dea_method'])
            wb = pd.read_excel(excel_file)
            colunas = list(wb.columns)
            for i in range(len(colunas)):
                colunas[i] = colunas[i].strip().lower()
            wb.columns = colunas
            colunas = list(wb.columns)
            inputs_dea = []
            outputs_dea = []
            print(wb.info())
            for i in range(len(colunas)):
                if i == 0:
                    indice_wb = colunas[i] 
                    continue
                elif 'i -' in colunas[i]:
                    inputs_dea.append(colunas[i])
                elif 'o -' in colunas[i]:
                    outputs_dea.append(colunas[i])
            wb.index = wb[f'{indice_wb}']
            wb = wb.drop(columns=[f'{indice_wb}'])
            print(wb.info())
            
            uni_prob = DEAProblem(wb[inputs_dea], wb[outputs_dea], returns=dea_method)
            myresults = uni_prob.solve()

            print(myresults['Status'])
            wb['Status'] = myresults['Status']
            wb[f'Efficiency - DEA {dea_method}'] = myresults['Efficiency']
            df_pesos = myresults['Weights']
            col_df_pesos = ['weights'+ i for i in list(df_pesos.columns) ]
            df_pesos.columns = col_df_pesos
            df_pesos.index = wb.index
            wb_ = wb.merge(myresults['Weights'], how='inner',left_index=True,right_index=True)
            response = HttpResponse(
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = f'attachment; filename="resultados_dea_{dea_method}.xlsx"'
            wb_.to_excel(response)
            return response
    else:
        form = UploadFileForm()
    return render(request, 'deaanalise.html', {'form': form})