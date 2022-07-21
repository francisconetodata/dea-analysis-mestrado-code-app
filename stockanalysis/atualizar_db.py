

def atualizar_db():
    import datetime
    import sqlite3  # Pacote do banco de dados

    import numpy as np
    import pandas as pd
    #from pandas.tseries.offsets import DateOffset
    from yahoo_fin.stock_info import get_data
    conn = sqlite3.connect('db.sqlite3')
    # Definindo um cursos para executar as funções.

    df_atualizar = pd.read_sql(f"""
                            SELECT ss.symbol_stock, ss.id, 
                            DATE(max(p.date_ref) , '+5 days') as date_ref 
                            FROM stockanalysis_stock ss
                            INNER JOIN stockanalysis_pricedatastocks p ON p.stock_id = ss.id
                            GROUP BY ss.symbol_stock, ss.id
                            """, conn)
    #df_atualizar['date_ref'] = pd.to_datetime(df_atualizar['date_ref'])
    print(df_atualizar.info())

    if False:
        for linha in cursor.fetchall():
            linha[1]
            dados = (get_data(linha[0])['adjclose'])
            dados = dados.reset_index()
            dados.columns = ['data', 'adj']
            for j in range(len(dados)):
                cursor.execute(f"""
                INSERT INTO stockanalysis_pricedatastocks (date_ref ,price_close, stock_id)
                VALUES ('{dados['data'][j]}','{abs(round(dados['adj'][j],2))}', {linha[1]} )
                """)
                conn.commit()
    if False:

        dados = (get_data('^BVSP')['adjclose'])
        dados = dados.reset_index()
        dados.columns = ['data', 'adj']
        for j in range(len(dados)):
            cursor.execute(f"""
            INSERT INTO stockanalysis_pricedatastocks (date_ref ,price_close, stock_id)
            VALUES ('{dados['data'][j]}','{abs(round(dados['adj'][j],2))}', 764 )
            """)
            conn.commit()

    for i in list(df_atualizar['symbol_stock'].values):
        cursor = conn.cursor()
        id_empresa = df_atualizar[df_atualizar['symbol_stock']
                                  == i]['id'].values[0]
        data_ref_db = df_atualizar[df_atualizar['symbol_stock']
                                   == i]['date_ref'].values[0]
        if (datetime.datetime.strptime(data_ref_db+" 22:22:22", '%Y-%m-%d %H:%M:%S') > datetime.datetime.now()) & (datetime.datetime.now().hour > 22):
            print(i, ' não foi atualizada! ')
            continue
        try:
            dados_atualizar = get_data(i,
                                       start_date=data_ref_db)['adjclose']
        except:
            print(i, ' não foi atualizada! ')
            continue
        else:
            dados_atualizar = get_data(i,
                                       start_date=data_ref_db)['adjclose']
        if len(dados_atualizar) == 0:
            print(i, ' não foi atualizada! ')
            continue

        dados_atualizar = dados_atualizar.reset_index()
        dados_atualizar.columns = ['data', 'preco']
        if dados_atualizar['data'].max() < datetime.datetime.strptime(data_ref_db, '%Y-%m-%d'):
            print(i, ' não será atualizada! ')
            continue
        else:
            for j in range(len(dados_atualizar)):
                cursor.execute(f"""
                INSERT INTO stockanalysis_pricedatastocks (date_ref ,price_close, stock_id)
                VALUES ('{dados_atualizar['data'][j]}','{abs(round(dados_atualizar['preco'][j],2))}', {id_empresa} )
                """)
                conn.commit()
            print(i, ' foi atualizada! ')


if __name__ == '__main__':
    atualizar_db()
