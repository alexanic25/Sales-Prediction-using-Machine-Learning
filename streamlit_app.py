import streamlit as st
import pandas as pd
import numpy as np

from streamlit_echarts import st_echarts
import plotly.graph_objects as go
from pyecharts.charts import Bar
from pyecharts import options as opts
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Predicție 2021")
st.cache(persist=True)

#prima parte: baza de date
st.subheader("Încărcare set de date")
df_name= st.selectbox("Alege setul de date", ('Inițial','Machine Learning'))

def get_dataset(df_name):
    path = 'D:/~SEM 2/Licenta/aplicatie VTEX/streamlit'
    if df_name == 'Inițial':
        df = pd.read_csv(path + '/df_complete.csv')
    elif df_name == 'Machine Learning':
        df = pd.read_csv(path + '/ml_dataset.csv')
    return df

if st.checkbox('Dimensiune set', key='1'):
    st.write("Număr de linii și coloane:", get_dataset(df_name).shape)
st.dataframe(get_dataset(df_name))

#partea a doua: vizualizare date
st.subheader('Vizualizare date')
viz_name= st.selectbox("Alege graficul", ('Hartă', 'Status', 'Comenzi lunare'))

##harta
data_init = pd.read_csv('D:/~SEM 2/Licenta/aplicatie VTEX/streamlit/df_complete.csv')
data_ml = pd.read_csv('D:/~SEM 2/Licenta/aplicatie VTEX/streamlit/ml_dataset.csv')

country_codes_dict = {}
for i in data_init['store_country'].value_counts().index:
  country_codes_dict[i] = str(i)[:2]

nb_app = data_init.groupby(['store_country'])['id'].count()
nb_app = nb_app.rename('nb_app').reset_index()


def get_map():
    fig = go.Figure(data=go.Choropleth(
        locations=nb_app['store_country'], 
        text=nb_app['store_country'],
        z = nb_app['nb_app'],
        colorscale = 'Reds',
        colorbar_title = 'Număr observații pe țară',
    ))

    fig.update_layout(
        title = dict(
        text = 'Harta țărilor cu prezență VTEX',
        font = dict(size = 20)
        ),
    )

    # Dimensiune harta
    fig.update_layout(height=550, width=900)
    st.plotly_chart(fig)

##status
counts = data_init['status'].value_counts()
percentages = counts * 100 / counts.sum()
percentages_wanted = percentages[percentages > 2]
percentages_wanted['others'] = percentages[percentages <= 2].sum()
result_dict = dict(zip(percentages_wanted.index, percentages_wanted.values))

def get_status():
    data = []
    for key, value in result_dict.items():
        data.append({"name": key, "value": round(value,2)})
    options = {
            'title': {'text': 'Statusul comenzilor'},
            'tooltip': {'trigger': 'item', 'formatter': '{a} <br/>{b}: {d} %'},
            'legend': {'right': 'right', 'data': list(result_dict.keys())},
            'series': [{'data': data, 'type': 'pie', 'name': 'Status'}]
            }
    st_echarts(options=options, height="400px")

## barchart comenzi pe luna

#df cu comenzi lunare
data_init['month_year'] = pd.to_datetime(pd.to_datetime(data_init['creation_date']).dt.month.astype(str) + '-' 
                                         + pd.to_datetime(data_init['creation_date']).dt.year.astype(str), format='%m-%Y')
monthly_orders = data_init.groupby('month_year')['id'].count()
monthly_orders = monthly_orders.rename('monthly_orders').reset_index()

# generare luni din 2021 in 2023
start_date = '2021-01-01'
end_date = '2023-03-31'
months_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%m-%Y')

# creare df
df_month_year = pd.DataFrame({'month_year': months_range})
df_month_year['month_year'] = pd.to_datetime(pd.to_datetime(df_month_year['month_year']).dt.month.astype(str) + '-' + pd.to_datetime(df_month_year['month_year']).dt.year.astype(str), format='%m-%Y')

# Concatenare cu monthly si 0 pentru unde nu exista, sortare dupa data
df_merged = pd.concat([df_month_year, monthly_orders], axis=0, ignore_index=True)
df_merged['monthly_orders'] = df_merged['monthly_orders'].fillna(0)
df_merged = df_merged.sort_values('month_year')

bar_x1 = list(df_merged['month_year'].astype(str))
bar_y1 = list(df_merged['monthly_orders'])

def get_bar(bar_x, bar_y, title, y_title):
    bar_chart = (Bar()
        .add_xaxis(bar_x)
        .add_yaxis(y_title, bar_y)
        .set_global_opts(title_opts=opts.TitleOpts(title=title),
                        toolbox_opts=opts.ToolboxOpts(),
                        yaxis_opts=opts.AxisOpts(min_ = 0, max_ = 9000))
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False)
        )
        .render_embed()
    )
    
    components.html(bar_chart, width=700, height=500)

def get_viz(viz_name):
    if viz_name == 'Status':
        get_status()
    elif viz_name == 'Hartă':
        get_map()
    elif viz_name == 'Comenzi lunare':
        get_bar(bar_x1, bar_y1, 'Comenzi lunare 2020-2023', 'Număr comenzi lunare')

get_viz(viz_name)

#partea a treia: selectare trasaturi
st.subheader('Selectare trăsături')
selection_type= st.selectbox("Alege modul de selectare al trăsăturilor:", ('Corelație Pearson', 'Selecție iterativă'))

X_train_init, y_train_init = data_ml[data_ml['year'] == 2020].drop('value', axis = 1),data_ml[data_ml['year']==2020]['value']
X_test_init, y_test_init = data_ml[data_ml['year'] == 2021].drop('value', axis = 1),data_ml[data_ml['year']==2021]['value']

dict = {'Random Forest Regressor': ['order_group_id', 'order_seller', 'account_group', 'ppp_value', 'store_country_BRA'], 
        'Extreme Gradient Boosting Regressor': ['order_seller', 'account_group', 'is_pandemic', 'unemployment_value', 'store_country_BRA'],
        'KNeighbors Regressor': ['order_group_id', 'tavg', 'ppp_value', 'unemployment_value', 'account_encoded'],
        'Linear Regression': ['order_group_id', 'order_seller', 'store_country_BRA', 'store_country_CHL', 'store_country_MEX']}

model_class = {'Random Forest Regressor':RandomForestRegressor(n_estimators = 500), 
               'Extreme Gradient Boosting Regressor':xgb.XGBRegressor(gamma = 1,n_estimators = 5000 ,max_depth = 5, learning_rate = 0.001),
               'KNeighbors Regressor': KNeighborsRegressor(n_neighbors = 8),
               'Support Vector Regressor': SVR(C = 10, kernel = 'linear'),
               'Linear Regression': LinearRegression()}

def get_forward(model_name, number_features,dict):
    features = []
    st.write('Cele mai importante trăsături pentru algoritm sunt:')
    for key, value in dict.items():
        if model_name == key:
            for i in range(1, number_features + 1):
                features.append(value[i-1])
                st.write(i, str(value[i - 1]))
            X_train_new, X_test_new = X_train_init[features], X_test_init[features]
    return X_train_new, X_test_new

def get_correlation(X_train_init, X_test_init, threshold):
  set_corr = set()
  corr_matrix_pearson = X_train_init.corr(method = 'pearson')
  for i in range(len(corr_matrix_pearson.columns)):
    for j in range(i):
      if abs(corr_matrix_pearson.iloc[i,j]) > threshold:
        set_corr.add(corr_matrix_pearson.columns[i])
  X_train_new,X_test_new = X_train_init.copy() ,X_test_init.copy()
  X_train_new.drop(columns = list(set_corr),inplace = True)
  X_test_new.drop(columns = list(set_corr),inplace = True)
  fig, ax = plt.subplots(figsize=(14,8))    
  sns.heatmap(X_train_new.drop('year',axis=1).corr(method = 'pearson'), cmap="coolwarm", annot=True, ax=ax, fmt=".3f")
  st.pyplot(fig)
  return X_train_new, X_test_new

def get_country(button0, button1, button2):
    if button0:
        return 0
    elif button1:
        return 1
    elif button2:
        return 2
    else:
        return -1

def train_test(model_name, X_train_new, X_test_new, X_test_init, y_train, y_test, button):
    for key, model in model_class.items():
        if key == model_name:
            if model_name == 'KNeighbors Regressor':
                pipeline = make_pipeline(StandardScaler(), model)
            else:
                pipeline = make_pipeline(None, model)
            pipeline.fit(X_train_new,y_train)
            y_pred_log = pipeline.predict(X_test_new)
            y_pred_org = np.expm1(y_pred_log)
            y_test_org = np.expm1(y_test)
            y_train_org = np.expm1(y_train)
            rmse = np.sqrt(mean_squared_error(y_test_org, y_pred_org))
            r2 = round(r2_score(y_test_org, y_pred_org),3)
            mae = round(mean_absolute_error(y_test_org, y_pred_org),3)
            rmse_norm = round(np.sqrt(mean_squared_error(y_test_org, y_pred_org))/np.mean(y_test_org),3)

            model_scores = pd.DataFrame(columns=['Model', 'RMSE', 'RMSE_norm','R2','MAE'])
            model_scores.loc[len(model_scores)] = [model_name, rmse, rmse_norm, r2, mae]
    df_test = X_test_init[['store_country_BRA', 'store_country_CHL','store_country_MEX']]
    df_train = X_train_init[['store_country_BRA', 'store_country_CHL','store_country_MEX']]
    column_mapping = {
        'store_country_BRA': 0,
        'store_country_CHL': 1,
        'store_country_MEX': 2
        }
    df_test = df_test.rename(columns=column_mapping)
    df_test_cat = df_test.idxmax(axis=1)
    df_train = df_train.rename(columns=column_mapping)
    df_train_cat = df_train.idxmax(axis=1)
    data_2020 = pd.DataFrame({'an': X_train_init['year'], 'valoare': y_train_org})
    data_2020['tara'] = df_train_cat
    data_2021 = pd.DataFrame({'an': X_test_init['year'], 'valoare': y_pred_org})
    data_2021['tara'] = df_test_cat
    result = pd.concat([data_2020, data_2021], axis=0, ignore_index=True)
    result = result.groupby(['an','tara'])['valoare'].sum()
    result = result.to_frame().reset_index()
    result['valoare'] = round(result['valoare'],0)
    result = result[result['tara'] == button]
    st.markdown('**Suma comenzilor pe an**')
    st.bar_chart(result, x = 'an', y = 'valoare')
    st.dataframe(model_scores)
    return round(r2*100,1)

def get_selection(selection_type):
    if selection_type == 'Selecție iterativă':
        col1, col2 = st.columns([1, 2])  # Adjust the column widths as needed

        with col1:
            number_features = st.number_input('Alege un numar de trasaturi', min_value = 1, max_value = 5, value = 5)

        with col2:
            model_name = st.selectbox("Alege algoritmul:", ('Random Forest Regressor', 'Extreme Gradient Boosting Regressor', 'KNeighbors Regressor', 'Linear Regression'))

        X_train_new, X_test_new = get_forward(model_name, number_features, dict)
    elif selection_type == 'Corelație Pearson':
        threshold = st.slider('Alegerea pragului:', 0.2, 1.0, 1.0, 0.1)
        X_train_new, X_test_new = get_correlation(X_train_init, X_test_init, threshold)
        model_name= st.selectbox("Alege modelul:", ('Random Forest Regressor', 'Extreme Gradient Boosting Regressor', 'KNeighbors Regressor', 'Support Vector Regressor', 'Linear Regression'))
    col1, col2, col3 = st.columns(3)
    with col1:
        b_button = st.button('Brazilia')
    with col2:
        c_button = st.button('Chile')
    with col3:
        m_button = st.button('Mexic')
    st.write('Modelul ales explică ', train_test(model_name, X_train_new, X_test_new, X_test_init, y_train_init, y_test_init, get_country(b_button,c_button,m_button)), '% din relația dintre variabilele independente și cea dependentă.')
    
get_selection(selection_type)