import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Weatherify dashboard",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #121212, #004D40, #191970);
        color: #FFFFFF;
    }
    
    .kpi-card {
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        height: 100%;
    }
    .kpi-title {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2.25rem;
        font-weight: 900;
        color: #1DB954;
    }
    .kpi-delta-pos {
        font-size: 1rem;
        font-weight: 500;
        color: #1DB954;
    }
    .kpi-delta-neg {
        font-size: 1rem;
        font-weight: 500;
        color: #FF4136;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


@st.cache_data
def generate_sample_data():
    date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
    songs = [{"track_name": "As It Was", "artist_names": "Harry Styles"}, {"track_name": "Anti-Hero", "artist_names": "Taylor Swift"}, {"track_name": "Flowers", "artist_names": "Miley Cyrus"}, {"track_name": "Calm Down", "artist_names": "Rema & Selena Gomez"}, {"track_name": "Escapism.", "artist_names": "RAYE"}, {"track_name": "Kill Bill", "artist_names": "SZA"}, {"track_name": "Miracle", "artist_names": "Calvin Harris, Ellie Goulding"}, {"track_name": "Vampire", "artist_names": "Olivia Rodrigo"}, {"track_name": "Sprinter", "artist_names": "Dave & Central Cee"}, {"track_name": "Paint The Town Red", "artist_names": "Doja Cat"}]
    
    data = []
    for date in date_range:
        day_of_year = date.timetuple().tm_yday
        temp_variation = 10 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        temperature = 10 + temp_variation + np.random.randn() * 2
        rain_prob = 0.2 + 0.15 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
        rain_sum = max(0, np.random.gamma(2, 2) - 2) if np.random.rand() < rain_prob else 0
        cloud_cover = np.clip(60 - 25 * np.sin(2 * np.pi * (day_of_year - 120) / 365) + np.random.uniform(-10, 10), 10, 95)
        sunshine_duration = np.clip((100 - cloud_cover) / 100 * 10 + np.random.uniform(-1, 1), 0, 12)
        song = songs[day_of_year % len(songs)]
        
        danceability = 0.65 + (temperature - 10) * 0.005 - (rain_sum * 0.01) + np.random.randn() * 0.05
        energy = 0.7 + (temperature - 10) * 0.004 + (sunshine_duration / 10) * 0.05 + np.random.randn() * 0.05
        valence = 0.5 + (sunshine_duration / 10) * 0.1 - (cloud_cover / 100) * 0.08 + np.random.randn() * 0.05
        
        data.append({"chart_date": date, "rank": 1, "track_name": song["track_name"], "artist_names": song["artist_names"], "danceability": np.clip(danceability, 0.4, 0.9), "energy": np.clip(energy, 0.4, 0.9), "valence": np.clip(valence, 0.2, 0.9), "tempo": 120 + np.sin(day_of_year / 20) * 5 + np.random.randn() * 3, "instrumentalness": np.clip(0.001 + np.random.rand() * 0.01, 0, 0.05), "temperature_2m_mean (°C)": temperature, "rain_sum (mm)": rain_sum, "sunshine_duration (h)": sunshine_duration})
        
    return pd.DataFrame(data)

df = generate_sample_data()

@st.cache_resource
def train_model(target_variable):
    features = ['temperature_2m_mean (°C)', 'rain_sum (mm)', 'sunshine_duration (h)']
    X = df[features]
    y = df[target_variable]
    model = LinearRegression()
    model.fit(X, y)
    return model

st.title("Weatherify dashboard")
st.markdown("<p style='color: rgba(255, 255, 255, 0.7);'>Explore the correlation between weather conditions in the UK and Spotify song features.</p>", unsafe_allow_html=True)

with st.expander("Select date range"):
    date_range = st.date_input(
        "Filter data by date",
        value=(df['chart_date'].min().date(), df['chart_date'].max().date()),
        min_value=df['chart_date'].min().date(),
        max_value=df['chart_date'].max().date(),
    )

if len(date_range) == 2:
    filtered_df = df[
        (df['chart_date'].dt.date >= date_range[0]) & 
        (df['chart_date'].dt.date <= date_range[1])
    ]
else:
    filtered_df = df

avg_valence = filtered_df['valence'].mean()
avg_energy = filtered_df['energy'].mean()
avg_temp = filtered_df['temperature_2m_mean (°C)'].mean()

delta_valence = avg_valence - df['valence'].mean()
delta_energy = avg_energy - df['energy'].mean()
delta_temp = avg_temp - df['temperature_2m_mean (°C)'].mean()

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average song positivity (valence)</div>
        <div class="kpi-value">{avg_valence:.3f}</div>
        <div class="{'kpi-delta-pos' if delta_valence >= 0 else 'kpi-delta-neg'}">
            {delta_valence:+.2%} vs. yearly average
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average song energy</div>
        <div class="kpi-value">{avg_energy:.3f}</div>
        <div class="{'kpi-delta-pos' if delta_energy >= 0 else 'kpi-delta-neg'}">
            {delta_energy:+.2%} vs. yearly average
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average temperature</div>
        <div class="kpi-value">{avg_temp:.1f}°C</div>
        <div class="{'kpi-delta-pos' if delta_temp >= 0 else 'kpi-delta-neg'}">
            {delta_temp:+.1f}°C vs. yearly average
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    with st.container(border=True):
        st.subheader("Trends over time")
        
        feature_options = {
            'valence': 'Song Positivity (Valence)',
            'energy': 'Song Energy',
            'danceability': 'Song Danceability'
        }
        selected_feature = st.selectbox(
            'Select a feature to display:',
            options=list(feature_options.keys()),
            format_func=lambda x: feature_options[x]
        )

        area_chart = alt.Chart(filtered_df).mark_area(
            line={'color':'#1DB954'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='rgba(29, 185, 84, 0.5)', offset=0),
                       alt.GradientStop(color='rgba(29, 185, 84, 0)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('chart_date:T', title='Date', axis=alt.Axis(labelColor='white', titleColor='white')),
            y=alt.Y(f'{selected_feature}:Q', title=feature_options[selected_feature], axis=alt.Axis(labelColor='white', titleColor='white')),
            tooltip=['chart_date', 'track_name', selected_feature]
        ).properties(
            background='transparent'
        ).configure_view(
            strokeOpacity=0
        ).interactive()
        
        st.altair_chart(area_chart, use_container_width=True)
        st.info("Insight: Audio features often show seasonal patterns, peaking during sunnier months and dipping in winter.")

with col2:
    with st.container(border=True):
        st.subheader("Audio fingerprint")
        radar_features = ['danceability', 'energy', 'valence', 'instrumentalness']
        
        sunny_data = df[df['sunshine_duration (h)'] > df['sunshine_duration (h)'].quantile(0.8)].mean(numeric_only=True)
        rainy_data = df[df['rain_sum (mm)'] > df['rain_sum (mm)'].quantile(0.8)].mean(numeric_only=True)

        sunny_values = [sunny_data[f] for f in radar_features]
        rainy_values = [rainy_data[f] for f in radar_features]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=sunny_values, theta=radar_features, fill='toself', name='Sunny Days', line=dict(color='#FFD700')))
        fig.add_trace(go.Scatterpolar(r=rainy_values, theta=radar_features, fill='toself', name='Rainy Days', line=dict(color='#1E90FF')))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)", 
                angularaxis=dict(linecolor='rgba(255,255,255,0.4)', gridcolor='rgba(255,255,255,0.2)', tickfont=dict(color='white')), 
                radialaxis=dict(showticklabels=False, linecolor='rgba(255,255,255,0.4)', gridcolor='rgba(255,255,255,0.2)')
            ), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            showlegend=True, 
            legend=dict(font=dict(color="white"), yanchor="bottom", y= -0.3, xanchor="center", x=0.5), 
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Insight: Music on sunny days tends to have higher energy and positivity (valence) compared to rainy days.")

st.markdown("---")

st.header("Explore correlations")
with st.container(border=True):
    corr_col1, corr_col2 = st.columns([1, 2])

    with corr_col1:
        st.subheader("Graph options")
        weather_options = {
            'temperature_2m_mean (°C)': 'Temperature (°C)',
            'rain_sum (mm)': 'Rainfall (mm)',
            'sunshine_duration (h)': 'Sunshine (hours)'
        }
        music_options = {
            'danceability': 'Danceability',
            'energy': 'Energy',
            'valence': 'Valence'
        }

        x_axis_var = st.selectbox(
            "Select weather variable (X-axis):",
            options=list(weather_options.keys()),
            format_func=lambda x: weather_options[x]
        )

        y_axis_var = st.selectbox(
            "Select music feature (Y-axis):",
            options=list(music_options.keys()),
            format_func=lambda x: music_options[x]
        )
    
    with corr_col2:
        scatter_plot = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.5, color='#1DB954').encode(
            x=alt.X(f'{x_axis_var}:Q', title=weather_options[x_axis_var]),
            y=alt.Y(f'{y_axis_var}:Q', title=music_options[y_axis_var]),
            tooltip=['chart_date', 'track_name', x_axis_var, y_axis_var]
        )

        regression_line = scatter_plot.transform_regression(
            x_axis_var, y_axis_var
        ).mark_line(color='white', strokeDash=[5,5])

        final_chart = (scatter_plot + regression_line).properties(
            background='transparent'
        ).configure_axis(
            labelColor='white',
            titleColor='white',
            gridColor='rgba(255, 255, 255, 0.2)'
        ).configure_view(
            strokeOpacity=0
        ).interactive()

        st.altair_chart(final_chart, use_container_width=True)

st.markdown("---")

st.header("Prediction playground")
st.markdown("<p style='color: rgba(255, 255, 255, 0.7);'>Use the sliders to simulate weather conditions and predict the musical vibe.</p>", unsafe_allow_html=True)

with st.container(border=True):
    pred_col1, pred_col2 = st.columns([1, 2])

    with pred_col1:
        st.subheader("Weather inputs")
        target_variable = st.selectbox(
            "Select music feature to predict:",
            ('valence', 'energy', 'danceability')
        )
        
        model = train_model(target_variable)

        input_temp = st.slider("Temperature (°C)", -5.0, 35.0, 15.0, 0.5)
        input_rain = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0, 1.0)
        input_sun = st.slider("Sunshine (hours)", 0.0, 15.0, 8.0, 0.5)

    with pred_col2:
        st.subheader("Predicted music vibe")
        input_data = pd.DataFrame({
            'temperature_2m_mean (°C)': [input_temp],
            'rain_sum (mm)': [input_rain],
            'sunshine_duration (h)': [input_sun]
        })
        
        prediction = model.predict(input_data)[0]

        st.metric(
            label=f"Predicted {target_variable.capitalize()}",
            value=f"{prediction:.3f}",
            help="This value is predicted by a linear regression model based on the weather inputs."
        )
        
        description = ""
        if target_variable == 'valence':
            description = 'positive and cheerful'
        elif target_variable == 'energy':
            description = 'energetic and intense'
        else:
            description = 'danceable and rhythmic'
            
        st.info(f"A higher **{target_variable}** score generally means the music is more {description}.")

