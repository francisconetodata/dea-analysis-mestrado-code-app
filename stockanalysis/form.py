
import datetime

from django import forms

from .models import Stock

#Stock = Stock.objects.order_by('name_stock')
class StockInput(forms.Form):
    CHOICES = Stock.objects.values_list('symbol_stock', 'name_stock').order_by('symbol_stock')
    data_inicial = forms.DateField(
        label='Data Incial',
        initial=(datetime.date.today()-datetime.timedelta(100)))
    data_final = forms.DateField(
        label='Data Final',
        initial=(datetime.date.today()-datetime.timedelta(20)))
    ticker = forms.ChoiceField(label='Empresa:',
                               choices=CHOICES)


class StockInputCart(forms.Form):
    CHOICES = Stock.objects.values_list('symbol_stock', 'name_stock').order_by('symbol_stock')

    data_inicial = forms.DateField(
        label='Data Incial',
        initial= datetime.date(2010,1,1),
        help_text='Formato: AAAA-MM-DD')
    data_final = forms.DateField(
        label='Data Final',
        initial= datetime.date(2020,1,1),
        help_text='Formato: AAAA-MM-DD')
    data_simulacao = forms.DateField(
        label='Data Simulação',
        initial= datetime.date(2022,1,1),
        help_text='Formato: AAAA-MM-DD')
    dea_method = forms.ChoiceField(label='Método DEA',
                                   choices=(('VRS', 'VRS'), ('CRS', 'CRS')))
    #selecionar_todas = forms.BooleanField(label='Selecionar todas as empresas presentes abaixo:')
    ticker = forms.MultipleChoiceField(label='Empresas listadas na bolsa (apenas setor elétrico):',
                                       widget=forms.CheckboxSelectMultiple,
                                       choices=CHOICES,required=False)

    risk_free = forms.FloatField(
        label='Taxa Livre de Risco (número absoluto):', initial=0.02,
        help_text='Utilize o separador decimal . (ponto). Corresponde ao Rf.')


class DownloadP(forms.Form):
    CHOICES = Stock.objects.values_list('symbol_stock', 'name_stock').order_by('symbol_stock')
    data_inicial = forms.DateField(
        label='Data Incial',
        initial= datetime.date(2010,1,1),
        help_text='Formato: AAAA-MM-DD')
    data_final = forms.DateField(
        label='Data Final',
        initial= datetime.date(2020,1,1),
        help_text='Formato: AAAA-MM-DD')
    ticker = forms.MultipleChoiceField(label='Empresa:',
                                       widget=forms.CheckboxSelectMultiple,
                                       choices=CHOICES)

class UploadFileForm(forms.Form):
    dea_method = forms.ChoiceField(label='Método DEA',
                                   choices=(('VRS', 'VRS'), ('CRS', 'CRS')))
    file = forms.FileField(label='file')