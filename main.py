import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

with st.echo(code_location='below'):
    second = pd.read_csv('TRADHIST_BITRADE_BITARIFF_2.txt', delimiter='.')
    third = pd.read_csv('TRADHIST_BITRADE_BITARIFF_3.txt', delimiter='.')
    tot = pd.concat([second, third], ignore_index=True, sort=True)
    tot=tot[tot['FLOW']==tot['FLOW']]
    tot['FLOW'] = tot['FLOW'].str.replace(',', '.').astype(float)
    st.write(f"## Проект по визуализации данных. Международная торговля в период с 1980г. по 2014г.")

    st.write(f"### Здесь можно увидеть динамику экспорта, импорта и чистого экспорта для выбранной страны.")
    country_din = st.multiselect("Выберите страну-экспортёра", pd.unique(tot["iso_o"]))
    n = st.slider('Выберите временной промежуток:', min_value=1980, max_value=2014, value=[1980, 2014], key=10)
    df_ex = tot[(tot['iso_o'] == country_din) & (tot['year'] >= n[0]) & (tot['year'] <= n[1])]
    df_ex = df_ex.groupby(by=df_ex.columns[4]).sum()
    df_im = tot[(tot['iso_d'] == country_din) & (tot['year'] >= n[0]) & (tot['year'] <= n[1])]
    df_im = df_im.groupby(by=df_im.columns[4]).sum()
    fig, ax = plt.subplots()
    sns.lineplot(x=df_ex.index, y=df_ex['FLOW'], ax=ax)
    st.pyplot(fig)
    fig, ax = plt.subplots()
    sns.lineplot(x=df_im.index, y=df_im['FLOW'], ax=ax)
    st.pyplot(fig)
    fig, ax = plt.subplots()
    sns.lineplot(x=df_im.index, y=np.array(df_ex['FLOW'])-np.array(df_im['FLOW']), ax=ax)
    st.pyplot(fig)

    st.write(f"### Здесь можно увидеть динамику двусторонних торговых потоков между заданными странами.")
    left_column, right_column = st.columns(2)
    country_1 = left_column.selectbox("Выберите страну-экспортёра", pd.unique(tot["iso_o"]), index=0, key=1)
    country_2 = right_column.selectbox("Выберите страну-импортёра", pd.unique(tot["iso_o"]), index=1, key=2)
    a = st.slider('Выберите временной промежуток:', min_value=1980, max_value=2014, value=[1980, 2014], key=11)
    df = tot[(tot['iso_o'] == country_1) & (tot['iso_d'] == country_2)]
    df = df[(df['year'] >= a[0]) & (df['year'] <= a[1])]
    y = df['FLOW'].astype(str).replace(',', '.').astype(float)
    fig, ax = plt.subplots()
    sns.lineplot(x=df['year'].astype(str).astype(float), y=y, ax=ax)
    st.pyplot(fig)

    st.write(f"### Посмотрим на то, насколько зависят выбранные страны друг от друга.")

    st.write(f"### Теперь перейдём к визуализации индекса открытости торговли.")
    st.write(f"### Индекс открытости торговли. Динамика.")

    st.write(f"### Индекс открытости торговли диаграмма рассеяния. Регрессия к ВВП (темпу роста ВВП?). Объём торговли vs ВВП.")

