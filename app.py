import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pycaret.regression import load_model, predict_model



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
    model = load_model('stc0_model_011')
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

def normalize(data):
    gen = 1
    spe = 1

    data['MinTimeToNearestStation'] = gen*((data['MinTimeToNearestStation']-nor_min['MinTimeToNearestStation'])/(nor_max['MinTimeToNearestStation']-nor_min['MinTimeToNearestStation']))

    data['Area'] = gen*((data['Area']-nor_min['Area'])/(nor_max['Area']-nor_min['Area']))
    
    data['Frontage'] = gen*((data['Frontage']-nor_min['Frontage'])/(nor_max['Frontage']-nor_min['Frontage']))
    
    data['BuildingYear'] = gen*((data['BuildingYear']-nor_min['BuildingYear'])/(nor_max['BuildingYear']-nor_min['BuildingYear']))
    
    data['CityPlanning'] = gen*((data['CityPlanning']-nor_min['CityPlanning'])/(nor_max['CityPlanning']-nor_min['CityPlanning']))
    
    data['Year'] = spe*((data['Year']-nor_min['Year'])/(nor_max['Year']-nor_min['Year']))

    data['Population'] = gen*((data['Population']-nor_min['Population'])/(nor_max['Population']-nor_min['Population']))

    data['GDP'] = gen*((data['GDP']-nor_min['GDP'])/(nor_max['GDP']-nor_min['GDP']))
    
    return data


# Global DATA:
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
nor_min = {
    'MinTimeToNearestStation': 0.0,
    'MaxTimeToNearestStation': 0.0,
    'Area': 10,
    'UnitPrice': 0,
    'Frontage': 0.0,
    'BuildingYear': 1945,
    'CityPlanning': 1,
    'Year': 2005,
    'Population': 557455,
    'GDP': 16978.0
}
nor_max = {
    'MinTimeToNearestStation': 120.0,
    'MaxTimeToNearestStation': 120.0,
    'Area': 2000,
    'UnitPrice': 56250000,
    'Frontage': 50.0,
    'BuildingYear': 2020,
    'CityPlanning': 16,
    'Year': 2019,
    'Population': 13940822,
    'GDP': 115682412.0
}
development_idx = {
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
    'Gunma':17,
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
    default='Tokyo',
    max_selections=1
)

# amount = st.sidebar.number_input('Investing amount:', step=100, value=1000000, min_value=0)
area = st.sidebar.number_input('Area (sq. feet):', step=1, value=90, min_value=1)
frontage = st.sidebar.number_input('Frontage Area (sq. feet):', step=1, value=5, min_value=0)
build_year = st.sidebar.number_input('Building Year:', step=1, value=1990)
min_time_to_station = st.sidebar.number_input('Minimum time to station (min):', step=1, value=15)
max_time_to_station = st.sidebar.number_input('Maximum time to station (min):', step=1, value=15)

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
    default='Quasi-residential Zone',
    max_selections=1
)
if cp:
    city_planning = cp_priority_dict[cp[0]]
else:
    city_planning = 2
year = st.sidebar.slider('Year:', 2024, 2030)

st.sidebar.markdown('#')
predict = st.sidebar.button('Predict')

model = get_model()

if predict:
    # to be fetch from time series predition:
    pop, gdp = get_dummy()

    holder = pd.DataFrame(columns=['Prefecture', 'MinTimeToNearestStation', 'Area', 
        'Frontage', 'BuildingYear', 'CityPlanning', 'Year', 'Population', 'GDP'])
    for i in range(2023, year+1):
        power = i - 2022
        holder.loc[len(holder.index)] = [prefecture[0], min_time_to_station, area, frontage, build_year, city_planning, i, get_pop(i, pop), get_gdp(i, gdp)]
    # st.dataframe(pred_holder)
    pred_holder = normalize(holder.copy(deep=True))

    progress_text = "Predicting and Preparing Graphs. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    pred_val = predict_model(model, data=pred_holder)
    for percent_complete in range(50):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 2, text=progress_text)
    my_bar.empty()
    # st.write(pred_val)

    per = ((pred_val['prediction_label'].values[-1] - pred_val['prediction_label'].values[0]) / pred_val['prediction_label'].values[0]) * 100
    st.subheader(f'Unit Price Estimates ({prefecture[0]}):')
    st.metric(label="PRICE (per unit area) wrt 2023:", value=f"{pred_val['prediction_label'].values[-1]:.2f} ¥", delta=f"{per:.2f}%")

 
    x = list(range(2023, year+1))
    y = pred_val['prediction_label'].values
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='Unit Price -->')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('##')
    lsec, rsec = st.columns(2)
    with lsec:
        st.subheader(f'GDP Estimate ({prefecture[0]}):')
        x = list(range(2023, year+1))
        y = holder['GDP'].values
        fig = go.Figure(data=go.Scatter(x=x, y=y))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='GDP -->')
        st.plotly_chart(fig, use_container_width=True)
    with rsec:
        st.subheader(f'Population Estimate ({prefecture[0]}):')
        x = list(range(2023, year+1))
        y = holder['Population'].values
        fig = go.Figure(data=go.Scatter(x=x, y=y))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Year -->', yaxis_title='Population -->')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader('Please provide the input details at the sidebar!')






    

    