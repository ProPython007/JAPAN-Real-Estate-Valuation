import time
import pickle
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
from tensorflow.keras.models import load_model



# Settings:
## Extra CSS:
st.set_page_config(page_title='Real Estate Investment Toolkit', page_icon=':bar_chart:', layout='wide')
hide_st_style = '''
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)


# Loading Resources:
@st.cache_resource
def get_model():
    model = load_model('stc0_tf_v13.h5')
    return model

@st.cache_data
def load_lat_long():
    return pd.read_csv('lat_lon.csv')

@st.cache_data
def get_gdp():
    return pd.read_csv('forecasted_gdp.csv')

@st.cache_data
def get_pop():
    return pd.read_csv('forecasted_pop.csv')

@st.cache_data
def get_aqi():
    return pd.read_csv('aqi.csv')

@st.cache_data
def get_cr():
    return pd.read_csv('cr.csv')


# Utility Function:
def load_gdp_forecast(pre):
    gname = f'GDP_FORECAST\\forecasts_{pre}.pkl'
    with open(gname, 'rb') as gforecast_file:
        gloaded_forecast = pickle.load(gforecast_file)
    return gloaded_forecast

def get_gdp_forecast(data, pre, y):
    return data[data['prefecture']==pre][str(y)].values[0]

def load_pop_forecast(pre):
    pname = f'POP_FORECAST\\forecasts_{pre}.pkl'
    with open(pname, 'rb') as pforecast_file:
        ploaded_forecast = pickle.load(pforecast_file)
    return ploaded_forecast

def get_pop_forecast(data, pre, y):
    return data[data['prefecture']==pre][str(y)].values[0]

def get_aqi_forecast(data, pre, y):
    return data[data['prefecture']==pre][str(y)].values[0]

def get_cr_forecast(data, pre, y):
    return data[data['prefecture']==pre][str(y)].values[0]

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}80'.format(r, g, b)

def gen_cols():
    red = 255
    green = 0
    stepSize = 11

    cols = []
    while (green < 255):
        green += stepSize
        if (green > 255):
            green = 255
        cols.append(rgb_to_hex(red, green, 0))

    while (red > 0):
        red -= stepSize
        if(red < 0):
            red = 0
        cols.append(rgb_to_hex(red, green, 0))

    return cols[:-1]

def get_plot_data():
    plot_data = load_lat_long()
    plot_data['amt'] = plot_data['prefecture'].values

    weights = {}
    for pre in pre_labelling:
        gloaded_forecast = load_gdp_forecast(pre)
        gloaded_forecast = gloaded_forecast.values[4:]
        gloaded_forecast = gloaded_forecast[:len(gloaded_forecast)-(2030-year)]
        gloaded_forecast = gloaded_forecast[-1]

        ploaded_forecast = load_pop_forecast(pre)
        ploaded_forecast = ploaded_forecast.values[4:]
        ploaded_forecast = ploaded_forecast[:len(ploaded_forecast)-(2030-year)]
        ploaded_forecast = ploaded_forecast[-1]

        pred_holder = pd.DataFrame(columns=['Prefecture', 'MinTimeToNearestStation', 'Area', 'Frontage', 'BuildingYear', 'CityPlanning', 'Year', 'Population', 'GDP'])
        pred_holder.loc[len(pred_holder.index)] = [pre_labelling[pre], min_time_to_station, area, frontage, build_year, city_planning, year, 0, 0]
        pred_holder['GDP'] = gloaded_forecast
        pred_holder['Population'] = ploaded_forecast

        tf_pred_holder = tf.constant(pred_holder, dtype=tf.float64)
        pred_val = model.predict(tf_pred_holder, verbose=0)[:]

        weights[pre] = pred_val[0][0]
    
    plot_data['amt'] = plot_data['amt'].apply(lambda x: weights[x])
    
    if plot_data['amt'].max() > 1000:
        plot_data['amt'] = plot_data['amt']/2000
    elif plot_data['amt'].max() > 500:
        plot_data['amt'] = plot_data['amt']/500
    elif plot_data['amt'].max() > 100:
        plot_data['amt'] = plot_data['amt']/100
    plot_data.sort_values(by=['amt'], inplace=True)
    plot_data['col'] = gen_cols()

    return plot_data

def gen_plot_data(pop_forecast, gdp_forecast):
    plot_data = load_lat_long()
    plot_data['amt'] = plot_data['prefecture'].values

    weights = {}
    for pre in pre_labelling:
        pred_holder = pd.DataFrame(columns=['Prefecture', 'MinTimeToNearestStation', 'Area', 'Frontage', 'BuildingYear', 'CityPlanning', 'Year', 'Population', 'GDP'])
        pred_holder.loc[len(pred_holder.index)] = [pre_labelling[pre], min_time_to_station, area, frontage, build_year, city_planning, year, get_pop_forecast(pop_forecast, pre, year), get_gdp_forecast(gdp_forecast, pre, year)]

        tf_pred_holder = tf.constant(pred_holder, dtype=tf.float64)
        pred_val = model.predict(tf_pred_holder, verbose=0)[:]

        weights[pre] = pred_val[0][0]
    
    plot_data['amt'] = plot_data['amt'].apply(lambda x: weights[x])
    
    if plot_data['amt'].max() > 1000:
        plot_data['amt'] = plot_data['amt']/2000
    elif plot_data['amt'].max() > 500:
        plot_data['amt'] = plot_data['amt']/500
    elif plot_data['amt'].max() > 100:
        plot_data['amt'] = plot_data['amt']/100
    plot_data.sort_values(by=['amt'], inplace=True)
    plot_data['col'] = gen_cols()
    plot_data.to_csv("file.csv",index=False)

    return plot_data



# Global DATA:
bank_avg_rate = 1.15
cp_priority_dict = {
    'Category I Exclusively Medium-high Residential Zone': 1,
    'Commercial Zone': 2,
    'Quasi-industrial Zone': 3,
    'Category I Residential Zone': 4,
    'Category I Exclusively Low-story Residential Zone': 5,
    'Quasi-residential Zone': 6,
    'Category II Residential Zone': 7,
    'Non-divided City Planning Area': 8,
    'Neighborhood Commercial Zone': 9,
    'Industrial Zone': 10,
    'Outside City Planning Area': 11,
    'Category II Exclusively Low-story Residential Zone': 12,
    'Urbanization Control Area': 13,
    'Category II Exclusively Medium-high Residential Zone': 14,
    'Exclusively Industrial Zone': 15,
    'Quasi-city Planning Area': 16
}

pre_labelling = {
    'Tokyo': 1,
    'Aichi': 2,
    'Shiga': 3,
    'Shizuoka': 4,
    'Fukui': 5,
    'Toyama': 6,
    'Osaka': 7,
    'Nagano': 8,
    'Ishikawa': 9,
    'Hiroshima': 10,
    'Kyoto': 11,
    'Mie': 12,
    'Kanagawa': 13,
    'Yamanashi': 14,
    'Okinawa': 15,
    'Kagawa': 16,
    'Gunma': 17,
    'Tochigi': 18,
    'Niigata': 19,
    'Hyogo': 20,
    'Oita': 21,
    'Gifu': 22,
    'Hokkaido': 23,
    'Ibaraki': 24,
    'Yamaguchi': 25,
    'Miyagi': 26,
    'Fukushima': 27,
    'Tottori': 28,
    'Shimane': 29,
    'Fukuoka': 30,
    'Kumamoto': 31,
    'Ehime': 32,
    'Chiba': 33,
    'Yamagata': 34,
    'Saga': 35,
    'Iwate': 36,
    'Tokushima': 37,
    'Nara': 38,
    'Saitama': 39,
    'Kochi': 40,
    'Wakayama': 41,
    'Miyazaki': 42,
    'Akita': 43,
    'Nagasaki': 44,
    'Kagoshima': 45,
    'Okayama': 46,
    'Aomori': 47
}


# MAIN:
st.title('Japan Real Estate Investment Prefecture wise Analysis')
st.markdown('###')

# __ SIDEBAR __
st.sidebar.header('Please Provide The Details Here:')
st.sidebar.markdown('#')

prefecture = st.sidebar.multiselect(
    'Select The Querry Prefecture:',
    options=['Fukui', 'Nagano', 'Nagasaki', 'Okayama', 'Fukushima', 'Mie',
       'Saitama', 'Wakayama', 'Ishikawa', 'Tokushima', 'Toyama', 'Ehime',
       'Tottori', 'Yamanashi', 'Yamaguchi', 'Saga', 'Miyazaki', 'Kyoto',
       'Hyogo', 'Tokyo', 'Kumamoto', 'Aichi', 'Kanagawa', 'Shizuoka',
       'Fukuoka', 'Oita', 'Yamagata', 'Akita', 'Chiba', 'Kochi',
       'Tochigi', 'Miyagi', 'Hokkaido', 'Okinawa', 'Iwate', 'Niigata',
       'Kagawa', 'Gunma', 'Ibaraki', 'Shiga', 'Nara', 'Gifu', 'Shimane',
       'Hiroshima', 'Osaka', 'Aomori', 'Kagoshima'],
    default='Okayama',
    max_selections=1
)

area = st.sidebar.number_input('Querry Area (sq. feet):', step=1, value=840, min_value=1)
frontage = st.sidebar.number_input('Querry Frontage Area (sq. feet):', step=1, value=12, min_value=0)
build_year = st.sidebar.number_input('Approx. Building Year:', step=1, value=1995)
min_time_to_station = st.sidebar.number_input('Approx. Minimum Time To Station (min):', step=1, value=15)

cp = st.sidebar.multiselect(
    'Select The Appropriate City Type:',
    options= ['Category I Exclusively Medium-high Residential Zone',
    'Commercial Zone',
    'Quasi-industrial Zone',
    'Category I Residential Zone',
    'Category I Exclusively Low-story Residential Zone',
    'Quasi-residential Zone',
    'Category II Residential Zone',
    'Non-divided City Planning Area',
    'Neighborhood Commercial Zone',
    'Industrial Zone',
    'Outside City Planning Area',
    'Category II Exclusively Low-story Residential Zone',
    'Urbanization Control Area',
    'Category II Exclusively Medium-high Residential Zone',
    'Exclusively Industrial Zone',
    'Quasi-city Planning Area'],
    default='Industrial Zone',
    max_selections=1
)
if cp:
    city_planning = cp_priority_dict[cp[0]]
else:
    city_planning = 0
year = st.sidebar.slider('Analysis Upto Year:', 2024, 2030, value=2028)

st.sidebar.markdown('#')
predict = st.sidebar.button('Predict')

model = get_model()

if predict:

    pop_forecast = get_pop()
    gdp_forecast = get_gdp()
    aqi_forecast = get_aqi()
    cr_forecast = get_cr()

    progress_text = "Predicting and Preparing Graphs. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(50):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 2, text=progress_text)
    my_bar.empty()

    plot_data = gen_plot_data(pop_forecast, gdp_forecast)
    st.subheader(f'Plot Price Estimates Over The Prefectures:')
    st.map(plot_data, latitude='lat', longitude='long', size='amt', color='col', use_container_width=True)
    
    st.sidebar.markdown('##')
    circle_style1 = "border-radius: 50%; width: 20px; height: 20px; background-color: red;"
    st.markdown(f'<div style="{circle_style1} display: inline-block;"></div> <div style="display: inline;">Investments with relatively low returns</div>', unsafe_allow_html=True)
    circle_style2 = "border-radius: 50%; width: 20px; height: 20px; background-color: green;"
    st.markdown(f'<div style="{circle_style2} display: inline-block;"></div> Investments with relatively high returns', unsafe_allow_html=True)
    
    st.markdown('##')
    st.markdown('##')

    with st.expander('Show Trend'):
        pred_holder = pd.DataFrame(columns=['Prefecture', 'MinTimeToNearestStation', 'Area', 'Frontage', 'BuildingYear', 'CityPlanning', 'Year', 'Population', 'GDP'])
        for i in range(2023, year+1):
            power = i - 2022
            pred_holder.loc[len(pred_holder.index)] = [prefecture[0], min_time_to_station, area, frontage, build_year, city_planning, i, get_pop_forecast(pop_forecast, prefecture[0], i), get_gdp_forecast(gdp_forecast, prefecture[0], i)]

        progress_text = "Predicting and Preparing Graphs. Please wait..."
        my_bar = st.progress(0, text=progress_text)

        pred_holder['Prefecture'] = pred_holder['Prefecture'].apply(lambda x : pre_labelling[x])
        tf_pred_holder = tf.constant(pred_holder, dtype=tf.float64)
        pred_val = model.predict(tf_pred_holder, verbose=0)[:]
        pred_holder['prediction_label'] = pred_val

        principal_val = pred_holder['prediction_label'].values[0]
        compounds = [principal_val]
        for i in range(2024, year+1):
            compounds.append(principal_val * (pow((1 + bank_avg_rate / 100), i-2023)))
        pred_holder['bank_interest'] = compounds

        for percent_complete in range(50):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 2, text=progress_text)
        my_bar.empty()

        st.markdown('##')

        st.dataframe(pred_holder[['Prefecture', 'Year', 'prediction_label', 'bank_interest']], use_container_width=True)

        per = ((pred_holder['prediction_label'].values[-1] - pred_holder['prediction_label'].values[0]) / pred_holder['prediction_label'].values[0])
        st.subheader(f'Plot Price Estimates ({prefecture[0]}):')
        st.metric(label="AVG PLOT PRICE wrt 2023:", value=f"{pred_holder['prediction_label'].values[-1]:.2f} ¥", delta=f"{per*100:.2f}%")

        x = list(range(2023, year+1))
        y = pred_holder['prediction_label'].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name='Real Estate Investment'))
        y = pred_holder['bank_interest'].values
        fig.add_trace(go.Scatter(x=x, y=y, name='Average Bank Returns', line_smoothing=1.3))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='Trade Price -->', showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('##')

        lsec, rsec = st.columns(2)

        with lsec:
            st.subheader(f'GDP Estimate ({prefecture[0]}):')
            x = list(range(2023, year+1))
            y = pred_holder['GDP'].values
            fig = go.Figure(data=go.Scatter(x=x, y=y, line_smoothing=1.3))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='GDP -->')
            st.plotly_chart(fig, use_container_width=True)

        with rsec:
            st.subheader(f'Population Estimate ({prefecture[0]}):')
            x = list(range(2023, year+1))
            y = pred_holder['Population'].values
            fig = go.Figure(data=go.Scatter(x=x, y=y, line_smoothing=1.3))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='Population -->')
            st.plotly_chart(fig, use_container_width=True)

        lsec2, rsec2 = st.columns(2)
        with lsec2:
            st.subheader(f'AQI Estimate ({prefecture[0]}):')
            x = list(range(2023, year+1))
            y = [get_aqi_forecast(aqi_forecast, prefecture[0], i) for i in range(2023, year+1)]
            fig = go.Figure(data=go.Scatter(x=x, y=y, line_smoothing=1.3))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='GDP -->')
            st.plotly_chart(fig, use_container_width=True)
        with rsec2:
            st.subheader(f'Crime Rate Estimate ({prefecture[0]}):')
            x = list(range(2023, year+1))
            y = [get_aqi_forecast(cr_forecast, prefecture[0], i) for i in range(2023, year+1)]
            fig = go.Figure(data=go.Scatter(x=x, y=y, line_smoothing=1.3))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='GDP -->')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader('Please provide the input details at the sidebar!')