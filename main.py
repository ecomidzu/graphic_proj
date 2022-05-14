import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from matplotlib.ticker import FormatStrFormatter
import plotly.express as px
from celluloid import Camera
import streamlit.components.v1 as components

with st.echo(code_location='below'):
    first = pd.read_csv('GDP.txt', delimiter='\t')
    second = pd.read_csv('TRADHIST_BITRADE_BITARIFF_2.txt', delimiter='.')
    third = pd.read_csv('TRADHIST_BITRADE_BITARIFF_3.txt', delimiter='.')
    tot = pd.concat([second, third], ignore_index=True, sort=True)
    tot = tot[tot['FLOW']==tot['FLOW']]
    tot['FLOW'] = tot['FLOW'].str.replace(',', '.').astype(float)
    st.write(f"## Проект по визуализации данных. Международная торговля в период с 1980г. по 2014г.")

    st.write('В рамках работы использованы стандартизированные iso3 коды для обозначения стран и образований из '
             'нескольких стран. Расшифровку можно посмотреть, например, '
             'здесь: https://ru.wikipedia.org/wiki/ISO_3166-1')
    st.write(f"### Здесь можно увидеть динамику экспорта и импорта для выбранной страны.")
    country_din = st.multiselect("Выберите страну-экспортёра", pd.unique(tot["iso_o"]), default=['JPN', 'CAN', 'USA'])
    n = st.slider('Выберите временной промежуток:', min_value=1980, max_value=2014, value=[1980, 2014], key=10)
    df_ex = tot[(tot['iso_o'].isin(country_din)) & (tot['year'] >= n[0]) & (tot['year'] <= n[1])]
    df_ex = df_ex.groupby(by=['iso_o', df_ex.columns[4]]).sum()
    df_im = tot[(tot['iso_d'].isin(country_din)) & (tot['year'] >= n[0]) & (tot['year'] <= n[1])]
    df_im = df_im.groupby(by=['iso_d', df_im.columns[4]]).sum()
    df_ex.index = df_ex.index.set_names(['Country', 'year'])
    df_ex.reset_index(inplace=True)
    df_im.index = df_im.index.set_names(['Country', 'year'])
    df_im.reset_index(inplace=True)

    fig1 = alt.Chart(df_ex).mark_line().encode(x=alt.X('year', axis=alt.Axis(title='Год')), y=alt.X('FLOW', axis=alt.Axis(title='ОБъём экспорта, долл. США')), color='Country')
    mark = alt.selection(type='single', nearest=True, on='mouseover', fields=['FLOW'])
    points = fig1.mark_circle().encode(opacity=alt.value(0)).add_selection(mark)
    text = fig1.mark_text(align='center', dy=10).encode(
       text=alt.condition(mark, 'FLOW', alt.value(' ')))
    lines = fig1.mark_line().encode(
        size=alt.condition(~mark, alt.value(0.5), alt.value(3))
    )
    fig2 = alt.Chart(df_im).mark_line().encode(x=alt.X('year', axis=alt.Axis(title='Год')), y=alt.X('FLOW', axis=alt.Axis(title='ОБъём импорта, долл. США')), color='Country')
    mark2 = alt.selection(type='single', nearest=True, on='mouseover', fields=['FLOW'])
    points2 = fig2.mark_circle().encode(opacity=alt.value(0)).add_selection(mark2)
    text2 = fig2.mark_text(align='center', dy=10).encode(
       text=alt.condition(mark2, 'FLOW', alt.value(' ')))
    lines2 = fig2.mark_line().encode(
        size=alt.condition(~mark2, alt.value(0.5), alt.value(3))
    )
    st.altair_chart(alt.layer(lines, points, text).interactive(), use_container_width=True)
    st.altair_chart(alt.layer(lines2, points2, text2).interactive(), use_container_width=True)

    st.write(f"### Здесь можно увидеть динамику двусторонних торговых потоков между заданными странами.")
    left_column, right_column = st.columns(2)
    country_1 = left_column.selectbox("Выберите страну-экспортёра", pd.unique(tot["iso_o"]), index=10, key=1)
    country_2 = right_column.selectbox("Выберите страну-импортёра", pd.unique(tot["iso_o"]), index=28, key=2)
    a = st.slider('Выберите временной промежуток:', min_value=1980, max_value=2014, value=[1980, 2014], key=11)
    df = tot[(tot['iso_o'] == country_1) & (tot['iso_d'] == country_2)]
    df = df[(df['year'] >= a[0]) & (df['year'] <= a[1])]
    y = df['FLOW'].astype(str).replace(',', '.').astype(float)
    fig = plt.figure()
    with sns.axes_style("whitegrid"):
        ax = fig.add_subplot()
    ax.set_title('Объём экспорта из ' + country_1 + ' в ' + country_2)
    ax.set_xlabel('Год')
    ax.set_ylabel('Объём торговли, доллары США')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    sns.set_style("whitegrid")

    sns.lineplot(x=df['year'].astype(str).astype(int), y=y, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write(
        f'### Посмотрим на распределение объёмов торговли выбранных стран.')
    c = list(pd.unique(tot["iso_o"]))[::4]
    countries_for_comp = st.multiselect("Выберите страны для сравнения", pd.unique(tot["iso_o"]), default=c)
    year = st.number_input('Введите год от 1980 до 2014: ', min_value=1980, max_value=2014, key=224)
    export = tot[(tot['year'] == year) & (tot['iso_o'].isin(countries_for_comp))]
    imports = tot[(tot['year'] == year) & (tot['iso_d'].isin(countries_for_comp))]
    dual = tot[
        (tot['year'] == year) & (tot['iso_d'].isin(countries_for_comp)) & (tot['iso_o'].isin(countries_for_comp))].drop(
        ['year', 'SOURCE_TF'], 1)
    export = export.groupby(by=['iso_o']).sum().drop('year', 1)
    imports = imports.groupby(by=['iso_d']).sum().drop('year', 1)
    new = export.join(imports, how='outer', lsuffix='_export', rsuffix='_import')
    new.fillna(0, inplace=True)
    new['total'] = new['FLOW_export'].astype('float') + new['FLOW_import'].astype('float')
    new.index = new.index.set_names(['Country'])
    new = new.sort_values(by=['total'], ascending = True)
    new.reset_index(inplace=True)
    fig = px.bar(new, x='total', y='Country',
                 hover_data=['FLOW_export', 'FLOW_import', 'total'], color='total',
                 labels={'total': 'долл. США ', 'FLOW_import': "Импорт, долл. США ", 'FLOW_export': "Экспорт, долл. США ",
                         'Country': 'Страна '}, height=1000, orientation='h', title='Внешнеторговый оборот')
    st.plotly_chart(fig)


    st.write(f"### Давайте посмотрим, как соотносятся доля экспорта и доля импорта в объёме торговли и ВВП на душу "
             f"населения.")
    st.write('Также добавим на график полиномиальную регрессию второй степени.')
    year_1 = st.number_input('Введите год от 1980 до 2014: ', min_value=1980, max_value=2014, key=223)
    @st.cache(allow_output_mutation=True)
    def get_data(year_1):
        trad = tot[tot['year'] == year_1].drop(['year', 'SOURCE_TF'], 1)
        export = trad.groupby(by=['iso_o']).sum()
        imports = trad.groupby(by=['iso_d']).sum()
        new = export.join(imports, how='outer', lsuffix='_export', rsuffix='_import')
        new.fillna(0, inplace=True)
        new['total'] = new['FLOW_export'].astype('float') + new['FLOW_import'].astype('float')
        new['exp_perc'] = new['FLOW_export'].astype('float') / new['total']
        new['imp_perc'] = new['FLOW_import'].astype('float') / new['total']
        GDP = first[['Country Code', str(year_1) + ' [YR' + str(year_1) + ']']].set_index('Country Code')
        GDP = new.join(GDP, how='inner')
        GDP.fillna(0)
        #GDP.columns = [0, 1, 2, 3, 4, year_1]
        GDP = GDP[(GDP[str(year_1) + ' [YR' + str(year_1) + ']'] != '..') & (GDP['exp_perc'] != 1) & (GDP['exp_perc'] != 0)]
        return GDP
    df1 = get_data(year_1)
    df1 = df1[df1[str(year_1) + ' [YR' + str(year_1) + ']'].astype(float) <= 50000]
    df1.fillna(0, inplace=True)
    fig1 = plt.figure()
    sf = plt.ScalarFormatter()
    sf.set_powerlimits((-2, 2))
    with sns.axes_style("dark"):
        ax3 = fig1.add_subplot()
        ax3.yaxis.set_major_formatter(sf)
    ax3.scatter(x=df1['exp_perc'], y=df1[str(year_1) + ' [YR' + str(year_1) + ']'].astype(float), s=4, c='g', marker='^')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.legend([str(year_1)], ncol=1, loc="upper left", bbox_to_anchor=(0.8,1))
    plt.yticks(rotation=45)
    plt.ylabel("ВВП на душу населения")
    plt.xlabel("Доля экспорта во внешнеторговом обороте")
    plt.title('Доля экспорта vs ВВП на душу населения, ' + str(year_1))
    X = np.array(df1['exp_perc'].astype(float))
    y = np.array(df1[str(year_1) + ' [YR' + str(year_1) + ']'].astype(float))
    model = np.poly1d(np.polyfit(X, y, 2))
    plt.plot(sorted(X), model(sorted(X)))
    st.pyplot(fig1)

    st.write(
        f"Здесь было бы интересно посмотреть на динамику формы полиномиальной регрессии сквозь года. Насколько вообще "
        f"сохраняется выпуклая вверх форма? Для этого создадим анимацию по годам.")
    fig2 = plt.figure()
    camera = Camera(fig2)
    with sns.axes_style("dark"):
        ax4 = fig2.add_subplot()
        ax4.yaxis.set_major_formatter(sf)
    for year in range(1980, 2015):
        df1 = get_data(year)
        df1 = df1[df1[str(year) + ' [YR' + str(year) + ']'].astype(float) <= 50000]
        df1.fillna(0, inplace=True)
        ax4.text(0.3, -0.13, 'Доля экспорта во внешнеторговом обороте', transform=ax.transAxes)
        ax4.text(-0.15, 0.15, 'ВВП на душу населения, доллары США', transform=ax.transAxes, rotation = 90)
        ax4.text(0.25, 1.01, 'Доля экспорта vs ВВП на душу населения', transform=ax.transAxes)
        ax4.scatter(x=df1['exp_perc'], y=df1[str(year) + ' [YR' + str(year) + ']'].astype(float), s=4, c='g',
                    marker='^')
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        plt.legend([str(year)], ncol=1, loc="upper left", bbox_to_anchor=(0.8, 1))
        X = np.array(df1['exp_perc'].astype(float))
        y = np.array(df1[str(year) + ' [YR' + str(year) + ']'].astype(float))
        model = np.poly1d(np.polyfit(X, y, 2))
        plt.plot(sorted(X), model(sorted(X)))
        camera.snap()
    animation = camera.animate()
    ### FROM: "https://gist.github.com/ischurov/fb00906c5704ebdd56ff13d7e02583e4"
    components.html(animation.to_jshtml(), height=1000)
    ### END FROM
    animation.save('animation.gif')

