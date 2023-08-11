import time
import pickle
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
from tensorflow.keras.models import load_model



# Settings:
## Extra CSS:
st.set_page_config(page_title='Real Estate Investment Predictor', page_icon=':bar_chart:', layout='wide')
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
def get_dummy():
    pop = pd.read_csv('pop_final.csv')
    gdp = pd.read_csv('gdp_final.csv')
    return pop, gdp


# Utility Function:
def get_pop(year, pop):
    grad_mult = year - 2019
    grad = pop[pop['Prefecture'] == prefecture[0]]['2019 Population'] - pop[pop['Prefecture'] == prefecture[0]]['2018 Population']
    population = int(pop[pop['Prefecture'] == prefecture[0]]['2019 Population']) + int(grad*grad_mult)
    return population

def get_gdp(year, gdp):
    grad_mult = year - 2019
    grad = gdp[gdp['Prefecture'] == prefecture[0]]['2019 GDP (in millions)'] - gdp[gdp['Prefecture'] == prefecture[0]]['2018 GDP (in millions)']
    gdp_val = int(gdp[gdp['Prefecture'] == prefecture[0]]['2019 GDP (in millions)']) + int(grad*grad_mult)
    return gdp_val



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
    'Select the Prefecture:',
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

area = st.sidebar.number_input('Area (sq. feet):', step=1, value=90, min_value=1)
frontage = st.sidebar.number_input('Frontage Area (sq. feet):', step=1, value=5, min_value=0)
build_year = st.sidebar.number_input('Building Year:', step=1, value=1990)
min_time_to_station = st.sidebar.number_input('Minimum time to station (min):', step=1, value=15)

cp = st.sidebar.multiselect(
    'Select the City Planning:',
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
year = st.sidebar.slider('Year:', 2024, 2030)

st.sidebar.markdown('#')
predict = st.sidebar.button('Predict')

model = get_model()

if predict:
    # to be fetch from time series predition:
    gname = f'GDP_FORECAST\\forecasts_{prefecture[0]}.pkl'
    with open(gname, 'rb') as gforecast_file:
        gloaded_forecast = pickle.load(gforecast_file)
    gloaded_forecast = gloaded_forecast.values[4:]
    gloaded_forecast = gloaded_forecast[:len(gloaded_forecast)-(2030-year)]

    pname = f'POP_FORECAST\\forecasts_{prefecture[0]}.pkl'
    with open(pname, 'rb') as pforecast_file:
        ploaded_forecast = pickle.load(pforecast_file)
    ploaded_forecast = ploaded_forecast.values[4:]
    ploaded_forecast = ploaded_forecast[:len(ploaded_forecast)-(2030-year)]
    # print(gloaded_forecast)
    # pop, gdp = get_dummy()

    pred_holder = pd.DataFrame(columns=['Prefecture', 'MinTimeToNearestStation', 'Area', 'Frontage', 'BuildingYear', 'CityPlanning', 'Year', 'Population', 'GDP'])
    for i in range(2023, year+1):
        power = i - 2022
        pred_holder.loc[len(pred_holder.index)] = [prefecture[0], min_time_to_station, area, frontage, build_year, city_planning, i, 0, 0]
    pred_holder['GDP'] = gloaded_forecast
    pred_holder['Population'] = ploaded_forecast
    # st.dataframe(pred_holder)

    progress_text = "Predicting and Preparing Graphs. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    pred_holder['Prefecture'] = pred_holder['Prefecture'].apply(lambda x : pre_labelling[x])
    tf_pred_holder = tf.constant(pred_holder, dtype=tf.float64)
    pred_val = model.predict(tf_pred_holder, verbose=0)[:]
    pred_holder['prediction_label'] = pred_val
    # st.write(pred_val)

    principal_val = pred_holder['prediction_label'].values[0]
    compounds = [principal_val]
    for i in range(2024, year+1):
        compounds.append(principal_val * (pow((1 + bank_avg_rate / 100), i-2023)))

    pred_holder['bank_interest'] = compounds
    
    for percent_complete in range(50):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 2, text=progress_text)
    my_bar.empty()


    per = ((pred_holder['prediction_label'].values[-1] - pred_holder['prediction_label'].values[0]) / pred_holder['prediction_label'].values[0]) * 100
    st.subheader(f'Plot Price Estimates ({prefecture[0]}):')
    st.metric(label="AVG PLOT PRICE wrt 2023:", value=f"{pred_holder['prediction_label'].values[-1]:.2f} ¥", delta=f"{per:.2f}%")


    x = list(range(2023, year+1))
    y = pred_holder['prediction_label'].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name='Real Estate Investment'))
    y = pred_holder['bank_interest'].values
    fig.add_trace(go.Scatter(x=x, y=y, name='Average Bank Returns'))
    # fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='Trade Price -->', showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # x = list(range(2023, year+1))
    # y = pred_holder['prediction_label'].values
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x, y=y, name='Graph 1'))
    # fig.add_trace(go.Scatter(x=x, y=y, name='Graph 2'))
    # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='Trade Price -->', showlegend=True)
    # st.plotly_chart(fig, use_container_width=True)


    st.markdown('##')
    lsec, rsec = st.columns(2)
    
    with lsec:
        st.subheader(f'GDP Estimate ({prefecture[0]}):')
        x = list(range(2023, year+1))
        y = pred_holder['GDP'].values
        fig = go.Figure(data=go.Scatter(x=x, y=y))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='GDP -->')
        st.plotly_chart(fig, use_container_width=True)
    
    with rsec:
        st.subheader(f'Population Estimate ({prefecture[0]}):')
        x = list(range(2023, year+1))
        y = pred_holder['Population'].values
        fig = go.Figure(data=go.Scatter(x=x, y=y))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='Population -->')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader('Please provide the input details at the sidebar!')