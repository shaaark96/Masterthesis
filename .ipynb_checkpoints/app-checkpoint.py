#!/usr/bin/env python
# coding: utf-8

# # Dashboard

# ### Pakete laden

# In[49]:


#get_ipython().system('pip install ipynbname')


# In[39]:


import pandas as pd
import pyproj
from dash import dcc, html, Input, Output
from jupyter_dash import JupyterDash
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.io as pio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import webbrowser
from sklearn.decomposition import PCA
import os
from dash import Dash
#import ipynbname


# ### Datein laden

# In[41]:


# ----------------------------
# Daten laden
# ----------------------------
df = pd.read_csv("data/merged_df.csv",  sep=';', low_memory=False)
#"C:\Users\sha_r\OneDrive - FH Graubünden\FHGR_\4. Semester\Masterthesis\dahsboard\data\merged_df.csv"

px.set_mapbox_access_token("pk.eyJ1Ijoic2hhcm9uMDgxNSIsImEiOiJjbWM4dXo0cmYxb3Q5MmpzM2xuNjAwcWo0In0.JsHM9h7Y2Sak3JWE1vH8lw")
pio.renderers.default = 'browser'

# ----------------------------
# Daten verarbeiten
# ----------------------------
df['time'] = pd.to_datetime(df['time'], errors='coerce')
time = df["time"].dropna().dt.strftime('%Y-%m-%d').unique()

unwetterarten = df["Unwetterart"].dropna().unique()

df.rename(columns={
    'Temp_Mittel': 'Temperatur_Mittelwert',
    'Temp_Abw': 'Temperatur_Abweichung',
    'Niederschlag_Max_10min': 'Niederschlag_Max_10min',
    'Niederschlag_Tag': 'Niederschlag_Tagessumme',
    'Sonnenschein_h': 'Sonnenscheindauer_Stunden',
    'Wind_kmh': 'Windgeschwindigkeit_kmh',
    'Bodentemp_100cm': 'Bodentemperatur_100cm',
    'Schweregrad' : 'Schadenskosten'    
}, inplace=True)

# Wettertypen (aus den numerischen Spalten)
wettertypen = ['Temperatur_Mittelwert', 'Temperatur_Abweichung', 'Niederschlag_Max_10min',
               'Niederschlag_Tagessumme', 'Sonnenscheindauer_Stunden', 'Windgeschwindigkeit_kmh', 'Bodentemperatur_100cm']

# Wetterstationen und Unwetterarten ieren
stationen = df["Name"].dropna().unique()

df['time'] = pd.to_datetime(df['time'], errors='coerce')


# ### App initialisieren

# In[43]:


# ----------------------------
# App-Initialisierung
# ----------------------------
app = Dash(__name__)
app.title = "Wetter Dashboard Schweiz"

# ----------------------------
# Layout-Funktion für Tabs
# ----------------------------
def create_tab_layout(tab_id):

    # Tab 1: Zeitverlauf – mehrere Graphen + Filter
    if tab_id == 1:
        return html.Div([
            html.Div([
                # Sidebar: Filterelemente
                html.Div([
                    html.Label("Zeitraum:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                    dcc.DatePickerRange(
                        id=f'date-picker-{tab_id}',
                        start_date=min(df["time"]),
                        end_date=max(df["time"]),
                        display_format='YYYY-MM-DD',
                        className="date-picker",
                        with_portal=True,
                        min_date_allowed=min(df["time"]),
                        max_date_allowed=max(df["time"]),
                        reopen_calendar_on_clear=True
                    ),
                    html.Button("Zurücksetzen", id=f'reset-date-{tab_id}', n_clicks=0, className="reset-button", style={"fontSize": "16px"}),
                ]),
                html.Label("Wettertyp:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id=f'wettertyp-{tab_id}',
                    options=[{'label': w, 'value': w} for w in wettertypen],
                    value="Temperatur_Mittelwert",
                    placeholder="Wettertyp wählen...",
                    style={"backgroundColor": "white", "color": "black","fontSize": "16px"}
                ),
                html.Label("Station:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id=f'station-{tab_id}',
                    options=[{'label': 'Alle Stationen', 'value': 'Alle'}] +
                            [{'label': s, 'value': s} for s in stationen],
                    placeholder="Station wählen...",
                    value="Alle",
                    style={"backgroundColor": "white", "color": "black","fontSize": "16px"}
                ),
                
                html.Label("Unwetterart:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id=f'unwetterart-{tab_id}',
                    options= [{'label': 'Alle Unwetterarten', 'value': 'Alle'}] + [{'label': w, 'value': w} for w in unwetterarten],
                    placeholder="Unwetterart wählen...",
                    value="Alle",
                    style={"backgroundColor": "white", "color": "black","fontSize": "16px"}
                )
            ], className="sidebar"),

            # Hauptinhalt: 4 Graphen
            html.Div([
                html.Div([
                    dcc.Graph(id='plot-1-top-left', className="graph-box"),
                    dcc.Graph(id='plot-1-top-right', className="graph-box")  # Scatter Mapbox oben rechts
                ], className="row-container"),

                html.Div([
                    dcc.Graph(id='plot-1-bottom-left', className="graph-box"),
                    dcc.Graph(id='plot-1-bottom-right', className="graph-box")  # Scatter Mapbox unten rechts
                ], className="row-container")
            ], className="content-area")
        ], className="main-layout")

    # Tab 2: Wetterstationen – Karte mit Stationen
    elif tab_id == 2:
        return html.Div([
            html.Div([
                # Sidebar: Zeitraum & Wettertyp
                html.Label("Zeitraum:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.DatePickerRange(
                    id='date-picker-2',
                    start_date=min(df["time"]),
                    end_date=max(df["time"]),
                    display_format='YYYY-MM-DD',
                    with_portal=True,
                    min_date_allowed=min(df["time"]),
                    max_date_allowed=max(df["time"]),
                    reopen_calendar_on_clear=True
                ),
                html.Label("Wettertyp:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id='wettertyp-2',
                    options=[{'label': w, 'value': w} for w in wettertypen],
                    value=wettertypen[0],
                    placeholder="Wettertyp wählen...",
                    style={"backgroundColor": "white", "color": "black","fontSize": "16px"}
                ),
            ], className="sidebar"),

            # Mapbox-Diagramm
            html.Div([
                dcc.Graph(
                    id='scatter-mapbox',
                    className="map-box",
                    style={"height": "600px", "width": "100%"}
                )
            ], className="content-area"),
        ], className="main-layout")

    # Tab 3: Naturereignis – Unwetterkarte
    elif tab_id == 3:
        return html.Div([
            html.Div([
                html.Label("Zeitraum:"),
                dcc.DatePickerRange(
                    id='date-picker-3',
                    start_date=min(df["time"]),
                    end_date=max(df["time"]),
                    display_format='YYYY-MM-DD',
                    with_portal=True,
                    min_date_allowed=min(df["time"]),
                    max_date_allowed=max(df["time"]),
                    reopen_calendar_on_clear=True
                ),
                html.Label("Unwetterart:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id='unwetterart-3',
                    options=[{'label': 'Alle Unwetterarten', 'value': 'Alle'}] +
                            [{'label': w, 'value': w} for w in unwetterarten],
                    value='Alle',
                    placeholder="Unwetterart wählen...",
                    style={"backgroundColor": "white", "color": "black","fontSize": "16px"}
                ),
                html.Label("Schadenskosten:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id='schadenskosten-3',    
                    options=[
                        {'label': 'Alle Schadenskosten', 'value': 'Alle'},
                        {'label': 'gering', 'value': 'gering'},
                        {'label': 'mittel', 'value': 'mittel'},
                        {'label': 'gross/katastrophal', 'value': 'gross/katastrophal'}
                    ],
                    value='Alle',
                    placeholder="Schadenskosten wählen...",
                    style={"backgroundColor": "white", "color": "black","fontSize": "16px"}
                ),
            ], className="sidebar"),

            # Mapbox-Diagramm
            html.Div([
                dcc.Graph(
                    id='scatter-mapbox-3',
                    className="map-box",
                    style={"height": "600px", "width": "100%"}
                )
            ], className="content-area"),
        ], className="main-layout")

    # Tab 4: Anomalie – Abweichungen visualisieren
    elif tab_id == 4:
        return html.Div([
            html.Div([
                html.Label("Zeitraum:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.DatePickerRange(
                    id='date-picker-4',
                    start_date=min(df["time"]),
                    end_date=max(df["time"]),
                    display_format='YYYY-MM-DD',
                    with_portal=True,
                    min_date_allowed=min(df["time"]),
                    max_date_allowed=max(df["time"]),
                    reopen_calendar_on_clear=True
                ),
                html.Label("Wettertyp:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id='wettertyp-4',
                    options=[{'label': w, 'value': w} for w in wettertypen],
                    value=wettertypen[0],
                    placeholder="Wettertyp wählen...",
                    style={"backgroundColor": "white", "color": "black","fontSize": "16px"}
                ),
                html.Label("Station:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}), 
                dcc.Dropdown(
                    id='station-4',
                    options=[{'label': 'Alle Stationen', 'value': 'Alle'}] +
                    [{'label': s, 'value': s} for s in stationen],
                    value="Alle",
                    placeholder="Station wählen...",
                    style={"backgroundColor": "white", "color": "black", "fontSize": "16px"}
                ),
            ], className="sidebar"),
            
            # Anomalie-Plot
            html.Div([
                dcc.Graph(
                    id='anomalie-plot',
                    className="map-box",
                    style={"height": "600px", "width": "100%"}
                )
            ], className="content-area"),
        ], className="main-layout")

    # Tab 5: Clustering – Clusteranalyse mit Mehrfachauswahl
    elif tab_id == 5:
        return html.Div([
            html.Div([
                html.Label("Zeitraum:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.DatePickerRange(
                    id='date-picker-5',
                    start_date=min(df["time"]),
                    end_date=max(df["time"]),
                    display_format='YYYY-MM-DD',
                    with_portal=True,
                    min_date_allowed=min(df["time"]),
                    max_date_allowed=max(df["time"]),
                    reopen_calendar_on_clear=True
                ),
                html.Label("Wettertyp:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id='wettertyp-5',
                    options=[{'label': w, 'value': w} for w in wettertypen],
                    value=["Temperatur_Mittelwert", "Niederschlag_Tagessumme"],  # Standardwert als Liste
                    multi=True,             # Mehrfachauswahl erlaubt
                    placeholder="Wettertyp wählen...",
                    style={
                        "backgroundColor": "white",
                        "color": "black",
                        "fontSize": "16px"
                    }
                ),
                html.Label("Anzahl Cluster:", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id='cluster-count-5',
                    options=[{'label': str(i), 'value': i} for i in range(2, 11)],
                    value=3,
                    clearable=False,
                    style={
                        "backgroundColor": "white",
                        "color": "black",
                        "fontSize": "16px"
                    }
                ),
            ], className="sidebar"),
            
            # Clustering-Diagramm
            html.Div([
                dcc.Graph(
                    id='clustering-plot',
                    className="map-box",
                    style={"height": "600px", "width": "100%"}
                )
            ], className="content-area"),
        ], className="main-layout")

    # Fallback / Standardinhalt
    else:
        return html.Div([
            html.H3("Inhalt folgt",  style={"color": "white"}),
            html.P("Hier folgt eine interaktive Karte", style={"color": "white"})
        ], className="content-area")

# ----------------------------
# Gesamtlayout der App inkl. Tabs
# ----------------------------
app.layout = html.Div([
    html.H1("Wetter- und Naturereignisdaten – Schweiz", className="dashboard-title", style={"fontSize": "42px"}),
    dcc.Tabs([
        dcc.Tab(label='Zeitverlauf', children=[create_tab_layout(1)], style={"fontSize": "18px"}),
        dcc.Tab(label='Wetterstationen', children=[create_tab_layout(2)], style={"fontSize": "18px"}),
        dcc.Tab(label='Unwetterereignis', children=[create_tab_layout(3)], style={"fontSize": "18px"}),
        dcc.Tab(label='Anomalie-Analyse', children=[create_tab_layout(4)], style={"fontSize": "18px"}),
        dcc.Tab(label='Clustering', children=[create_tab_layout(5)], style={"fontSize": "18px"}),
        dcc.Tab(label='Über das Projekt', style={"fontSize": "18px"}, children=[
            html.Div([
                html.H3("Über das Projekt", style={"color": "white", "fontSize": "22px", "fontWeight": "bold"}),
                html.P("""
                    Dieses interaktive Dashboard wurde im Rahmen einer Masterarbeit im Studiengang Data Visualization 
                    an der Fachhochschule Graubünden entwickelt. Ziel des Projekts ist es, Wetterdaten mit Naturereignissen 
                    in der Schweiz zu verknüpfen und visuell verständlich darzustellen.
                """, style={"color": "white"}),
                html.P("""
                    Die zugrundeliegenden Daten stammen von MeteoSchweiz sowie vom WSL – Institut für Schnee- und Lawinenforschung SLF.
                    Sie umfassen verschiedene Wetterparameter wie Temperatur, Niederschlag, Sonnenscheindauer und Windgeschwindigkeit 
                    sowie Daten zu Extremereignissen und deren Schadensauswirkungen.
                """, style={"color": "white"}),
                html.P("""
                    Die Analyse zielt darauf ab, Anomalien zu identifizieren und visuelle Muster zu erkennen, 
                    die auf potenzielle Zusammenhänge zwischen Wetterphänomenen und Naturgefahren hinweisen.
                """, style={"color": "white"}),
                html.P("""
                    Erstellt wurde das Dashboard von Sharion Reiser. Die wissenschaftliche Betreuung erfolgte durch 
                    Dr. Michael Burch.
                """, style={"color": "white"}),
                html.P("""
                    Die Visualisierungen sollen Fachpersonen, Forschenden sowie Entscheidungsträger:innen einen 
                    intuitiven Zugang zu komplexen Datensätzen ermöglichen und zur Risikoabschätzung und 
                    Prävention von Naturereignissen beitragen.
                """, style={"color": "white"})
            ])
        ])
    ])
])


# ----------------------------
# Callbacks
# ----------------------------

# --- Callbacks Tab 1 ---
@app.callback(
    Output('date-picker-1', 'start_date'),
    Output('date-picker-1', 'end_date'),
    Input('reset-date-1', 'n_clicks'),
    prevent_initial_call=True
)
def reset_date(n_clicks):
    return min(df["time"]), max(df["time"])

@app.callback(
    Output('plot-1-top-left', 'figure'),
    Output('plot-1-bottom-left', 'figure'),
    Input('wettertyp-1', 'value'),
    Input('station-1', 'value'),
    Input('date-picker-1', 'start_date'),
    Input('date-picker-1', 'end_date'),
    Input('unwetterart-1', 'value'),
)
def update_tab1_plots(wettertyp, station, start_date, end_date, unwetterart):
    empty_fig = px.bar(title="Bitte Filter auswählen")
    empty_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    if not wettertyp:
        return empty_fig, empty_fig

    dff = df.copy()
    dff['time'] = pd.to_datetime(dff['time'], errors='coerce')
    dff = dff[
        (dff['time'] >= pd.to_datetime(start_date)) &
        (dff['time'] <= pd.to_datetime(end_date))
    ]

    if station != "Alle":
        dff = dff[dff['Name'] == station]

    dff['Datum'] = dff['time'].dt.date
    df_agg = dff.groupby("Datum")[wettertyp].mean().reset_index()

    fig1 = px.line(
        df_agg,
        x="Datum", y=wettertyp,
        title=f"{wettertyp} pro Tag – {'alle Stationen' if station == 'Alle' else station}",
        color_discrete_sequence=["#2f2e88"]
    )
    fig1.update_layout(
        xaxis_title="Datum", yaxis_title=wettertyp,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    dff_uw = df[df["Unwetterart"].notna()].copy()
    dff_uw['time'] = pd.to_datetime(dff_uw['time'], errors='coerce')
    dff_uw['Datum'] = dff_uw['time'].dt.date
    dff_uw = dff_uw[
        (dff_uw['time'] >= pd.to_datetime(start_date)) &
        (dff_uw['time'] <= pd.to_datetime(end_date))
    ]

    if unwetterart and unwetterart != "Alle":
        dff_uw = dff_uw[dff_uw['Unwetterart'] == unwetterart]

    if not dff_uw.empty:
        if unwetterart and unwetterart != "Alle":
            unwetter_counts = dff_uw.groupby("Datum").size().reset_index(name="Anzahl")
            fig2 = px.bar(unwetter_counts, x="Datum", y="Anzahl",
                          title=f"{unwetterart} pro Tag",
                          color_discrete_sequence=["#2f2e88"])
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        else:
            unwetter_counts = dff_uw.groupby(["Datum", "Unwetterart"]).size().reset_index(name="Anzahl")
            farben = {
                "Rutschung": "#2f2e88",
                "Sturz": "#6b69d6",
                "Murgang": "#4b49a5",
                "Hochwasser": "#a9a9c5"
            }
            fig2 = px.bar(unwetter_counts, x="Datum", y="Anzahl", color="Unwetterart",
                          color_discrete_map=farben,
                          title="Unwetter pro Tag nach Art")
            fig2.update_layout(
                xaxis_title="Datum", yaxis_title="Anzahl",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                legend_title="Unwetterart"
            )
    else:
        fig2 = px.bar(title="Keine Unwetterdaten im Zeitraum vorhanden")
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

    return fig1, fig2


# --- Neuer Callback für die rechte Mapbox-Plots in Tab 1 ---
@app.callback(
    Output('plot-1-top-right', 'figure'),
    Output('plot-1-bottom-right', 'figure'),
    Input('date-picker-1', 'start_date'),
    Input('date-picker-1', 'end_date'),
    Input('wettertyp-1', 'value'),
    Input('unwetterart-1', 'value'),
)
def update_right_maps(start_date, end_date, wettertyp, unwetterart):

    def classify_change(delta, threshold):
        if delta < -threshold:
            return 'Abnahme'
        elif delta > threshold:
            return 'Zunahme'
        else:
            return 'Keine Veränderung'

    farben = {
        'Abnahme': 'blue',
        'Keine Veränderung': 'lightgray',
        'Zunahme': 'red'
    }

    # 1) DataFrame filtern
    dff = df.copy()
    dff['time'] = pd.to_datetime(dff['time'])
    dff = dff[(dff['time'] >= pd.to_datetime(start_date)) &
              (dff['time'] <= pd.to_datetime(end_date))]
    if dff.empty:
        empty = px.scatter_mapbox(title="Keine Daten im Zeitraum")
        return empty, empty

    # 2) In zwei Hälften splitten
    t0, t1 = dff['time'].min(), dff['time'].max()
    mid = t0 + (t1 - t0) / 2
    df1 = dff[dff['time'] <= mid]
    df2 = dff[dff['time'] >  mid]

    # 3) Oben rechts: Wettertyp
    if wettertyp:
        g1 = df1.groupby(['Name','lat','lon'])[wettertyp] \
               .mean().reset_index(name='wert1')
        g2 = df2.groupby(['Name','lat','lon'])[wettertyp] \
               .mean().reset_index(name='wert2')
        if g1.empty or g2.empty:
            fig_top = px.scatter_mapbox(title="Nicht genug Daten für Vergleich")
        else:
            diff = pd.merge(g1, g2, on=['Name','lat','lon'], how='inner')
            diff['delta'] = diff['wert2'] - diff['wert1']
            diff['delta_klasse'] = diff['delta'].apply(lambda x: classify_change(x, 0.1))

            fig_top = px.scatter_mapbox(
                diff, lat="lat", lon="lon",
                hover_name="Name",
                hover_data={"wert1":True, "wert2":True, "delta":True},
                color="delta_klasse",
                color_discrete_map=farben,
                zoom=7,
                title=f"Veränderung {wettertyp}"
            )

            fig_top.update_traces(marker=dict(size=20))  # feste Punktgröße
            
            fig_top.update_layout(
                mapbox_style="streets",
                legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
                legend_title="Veränderung",
                margin={"r":0,"t":40,"l":0,"b":0}
            )
    else:
        fig_top = px.scatter_mapbox(title="Kein Wettertyp gewählt")

    # 4) Unten rechts: Unwetter
    uw = dff[dff['Unwetterart'].notna()].copy()
    if uw.empty:
        return fig_top, px.scatter_mapbox(title="Keine Unwetterdaten")

    # Name-Fallback für NaN
    uw['Name'] = uw['Name'].fillna(uw['lat'].round(4).astype(str) + ", " + uw['lon'].round(4).astype(str))

    uw1 = uw[uw['time'] <= mid]
    uw2 = uw[uw['time'] >  mid]

    if unwetterart and unwetterart != "Alle":
        uw1 = uw1[uw1['Unwetterart'] == unwetterart]
        uw2 = uw2[uw2['Unwetterart'] == unwetterart]

    g1_uw = uw1.groupby(['Name','lat','lon']).size().reset_index(name='anz1')
    g2_uw = uw2.groupby(['Name','lat','lon']).size().reset_index(name='anz2')

    if g1_uw.empty and g2_uw.empty:
        fig_bottom = px.scatter_mapbox(title="Keine Unwetterdaten zum Vergleich")
    else:
        dfuw = pd.merge(g1_uw, g2_uw, on=['Name','lat','lon'], how='outer').fillna(0)
        dfuw['delta'] = dfuw['anz2'] - dfuw['anz1']
        dfuw['delta_klasse'] = dfuw['delta'].apply(lambda x: classify_change(x, 1))

        fig_bottom = px.scatter_mapbox(
            dfuw, lat="lat", lon="lon",
            hover_name="Name",
            hover_data={"anz1":True, "anz2":True, "delta":True},
            color="delta_klasse",
            color_discrete_map=farben,
            zoom=7,
            title=f"Veränderung Unwetteranzahl ({unwetterart})"
        )

        fig_bottom.update_traces(marker=dict(size=10))  # feste Punktgröße
        
        fig_bottom.update_layout(
            mapbox_style="streets",
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
            legend_title="Veränderung",
            margin={"r":0,"t":40,"l":0,"b":0}
        )

    return fig_top, fig_bottom


# --- Callback Tab 2 ---
@app.callback(
    Output('scatter-mapbox', 'figure'),
    Input('date-picker-2', 'start_date'),
    Input('date-picker-2', 'end_date'),
    Input('wettertyp-2', 'value'),
)
def update_scatter_map(start_date, end_date, wettertyp):
    if not start_date or not end_date or not wettertyp:
        return px.scatter_mapbox(title="Bitte Filter auswählen")

    dff = df.copy()
    dff = dff[(dff['time'] >= pd.to_datetime(start_date)) & (dff['time'] <= pd.to_datetime(end_date))]
    dff = dff.dropna(subset=[wettertyp])
    

    if dff.empty:
        return px.scatter_mapbox(title="Keine Daten im ausgewählten Zeitraum")

    # Mittelwerte pro Station
    grouped = dff.groupby(['Name', 'lat', 'lon'])[wettertyp].mean().reset_index()

    center_lat = grouped['lat'].mean() if not grouped['lat'].isnull().all() else 46.8
    center_lon = grouped['lon'].mean() if not grouped['lon'].isnull().all() else 8.2
    

    fig = px.scatter_mapbox(
        grouped,
        lat="lat",
        lon="lon",
        hover_name="Name",
        hover_data=[wettertyp],
        color=wettertyp,
        color_continuous_scale="Plasma",
        #range_color=[-max_abs, max_abs],
        zoom=7,
    )

    fig.update_traces(marker=dict(size=20))  # feste Punktgröße
    
    fig.update_layout(
        mapbox_style="streets",
        mapbox_center= {"lat": center_lat, "lon": center_lon},
        margin={"r":0, "t":40, "l":0, "b":0},
        title=f"Wetterparameter '{wettertyp}'",
        uirevision='fixed'
    )
    return fig


# --- Callback Tab 3 ---
@app.callback(
    Output('scatter-mapbox-3', 'figure'),
    Input('date-picker-3', 'start_date'),
    Input('date-picker-3', 'end_date'),
    Input('unwetterart-3', 'value'),
    Input('schadenskosten-3', 'value'),  
)
def update_scatter_map_tab3(start_date, end_date, unwetterart, schweregrad):
    if not start_date or not end_date:
        return px.scatter_mapbox(title="Bitte Zeitraum wählen")

    dff = df.copy()
    dff['time'] = pd.to_datetime(dff['time'], errors='coerce')
    dff = dff[
        (dff['time'] >= pd.to_datetime(start_date)) &
        (dff['time'] <= pd.to_datetime(end_date)) &
        dff['Unwetterart'].notna() & dff['lat'].notna() & dff['lon'].notna()
    ]

    if unwetterart != "Alle":
        dff = dff[dff['Unwetterart'] == unwetterart]

    if schweregrad != "Alle":
        dff = dff[dff['Schadenskosten'] == schweregrad]  

    if dff.empty:
        return px.scatter_mapbox(title="Keine Unwetterdaten im ausgewählten Zeitraum und Filter")

    fig = px.scatter_mapbox(
        dff,
        lat="lat",
        lon="lon",
        hover_name="Gemeinde",
        hover_data=["Unwetterart", "Schadenskosten", "time"],
        color="Schadenskosten", 
        color_discrete_map={"leicht": "green", "mittel": "yellow", "schwer": "red"},
        title=f"Unwetterkarte: {unwetterart} ({schweregrad})" if schweregrad != "Alle" else f"Unwetterkarte: {unwetterart}",
        zoom=7
    )

    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        mapbox_style="streets",
        mapbox_center={"lat": dff['lat'].mean(), "lon": dff['lon'].mean()},
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    return fig


# --- Callback Tab 4 ---
@app.callback(
    Output('anomalie-plot', 'figure'),
    Input('date-picker-4', 'start_date'),
    Input('date-picker-4', 'end_date'),
    Input('wettertyp-4', 'value'),
    Input('station-4', 'value'),
)
def update_anomaly_plot(start_date, end_date, wettertyp, station):
    dff = df.copy()
    dff['time'] = pd.to_datetime(dff['time'])
    dff = dff[(dff['time'] >= pd.to_datetime(start_date)) & (dff['time'] <= pd.to_datetime(end_date))]
    dff = dff.dropna(subset=[wettertyp, 'Name'])

    if station == "Alle":
        # Aggregierter Mittelwert mit Min/Max-Bereich
        grouped = dff.groupby('time')
        df_summary = grouped.agg(
            mean_val=(wettertyp, 'mean'),
            min_val=(wettertyp, 'min'),
            max_val=(wettertyp, 'max')
        ).reset_index()

        # Z-Score für Mittelwert
        mean = df_summary['mean_val'].mean()
        std = df_summary['mean_val'].std()
        df_summary['z'] = (df_summary['mean_val'] - mean) / std
        df_summary['anomalie'] = abs(df_summary['z']) > 2

        fig = go.Figure()

        # Bereich Min/Max
        fig.add_trace(go.Scatter(
            x=pd.concat([df_summary['time'], df_summary['time'][::-1]]),
            y=pd.concat([df_summary['max_val'], df_summary['min_val'][::-1]]),
            fill='toself',
            fillcolor='rgba(173, 216, 230, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Min/Max Bereich'
        ))

        # Mittelwertlinie
        fig.add_trace(go.Scatter(
            x=df_summary['time'],
            y=df_summary['mean_val'],
            mode='lines',
            name='Mittelwert',
            line=dict(color='blue')
        ))

        # Anomaliepunkte
        fig.add_trace(go.Scatter(
            x=df_summary['time'][df_summary['anomalie']],
            y=df_summary['mean_val'][df_summary['anomalie']],
            mode='markers',
            name='Anomalie',
            marker=dict(color='red', size=10, symbol='x')
        ))

        fig.update_layout(
            title=f"Anomalie-Erkennung ({wettertyp}) – Mittelwert aller Stationen",
            xaxis_title="Datum",
            plot_bgcolor='rgba(0,0,0,0)',  # Plot-Hintergrund transparent
            paper_bgcolor='rgba(0,0,0,0)',  # Gesamter Bereich (inkl. Ränder) transparent
            yaxis_title=wettertyp
        )
        return fig

    else:
        # Einzelstation: Verlauf + Anomalien
        df_station = dff[dff['Name'] == station].copy()

        mean = df_station[wettertyp].mean()
        std = df_station[wettertyp].std()
        df_station['z'] = (df_station[wettertyp] - mean) / std
        df_station['anomalie'] = abs(df_station['z']) > 2

        fig = go.Figure()

        # Verlauf
        fig.add_trace(go.Scatter(
            x=df_station['time'],
            y=df_station[wettertyp],
            mode='lines',
            name=f"{station}"
        ))

        # Anomalien
        fig.add_trace(go.Scatter(
            x=df_station['time'][df_station['anomalie']],
            y=df_station[wettertyp][df_station['anomalie']],
            mode='markers',
            name='Anomalie',
            marker=dict(color='red', size=10, symbol='x')
        ))

        fig.update_layout(
            title=f"Anomalien ({wettertyp}) – Station: {station}",
            plot_bgcolor='rgba(0,0,0,0)',  # Plot-Hintergrund transparent
            paper_bgcolor='rgba(0,0,0,0)',  # Gesamter Bereich (inkl. Ränder) transparent
            xaxis_title="Zeit",
            yaxis_title=wettertyp
        )
        return fig

# --- Callback Tab 5 ---
@app.callback(
    Output('clustering-plot', 'figure'),
    Input('date-picker-5', 'start_date'),
    Input('date-picker-5', 'end_date'),
    Input('wettertyp-5', 'value'),
    Input('cluster-count-5', 'value')
)
def perform_clustering(start_date, end_date, selected_features, n_clusters):
    def leere_karte(meldung="Bitte gültige Eingaben wählen"):
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
            title=meldung,
            mapbox_style="streets",
            mapbox_center={"lat": 46.8, "lon": 8.2},  # Fallback Zentrum (CH)
            mapbox_zoom=7,
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        return fig

    if not selected_features or len(selected_features) < 2:
        return leere_karte("Mindestens zwei Wettertypen wählen für Clustering")

    if not start_date or not end_date:
        return leere_karte("Bitte gültigen Zeitraum wählen")

    dff = df.copy()
    dff['time'] = pd.to_datetime(dff['time'], errors='coerce')
    dff = dff[
        (dff['time'] >= pd.to_datetime(start_date)) &
        (dff['time'] <= pd.to_datetime(end_date)) &
        dff['lat'].notna() & dff['lon'].notna()
    ]

    if dff.empty:
        return leere_karte("Keine Daten im gewählten Zeitraum")

    try:
        # Clustering vorbereiten
        X = dff[selected_features].dropna()
        coords = dff.loc[X.index, ['lat', 'lon', 'Name']]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        coords['Cluster'] = cluster_labels.astype(str)

        # Karten-Zentrum mit Fallback
        center_lat = coords['lat'].mean() if not coords['lat'].isnull().all() else 46.8
        center_lon = coords['lon'].mean() if not coords['lon'].isnull().all() else 8.2

        fig = px.scatter_mapbox(
            coords,
            lat='lat',
            lon='lon',
            color='Cluster',
            hover_name='Name',
            zoom=7,
            title=f'Clusteranalyse ({n_clusters} Cluster)'
        )

        fig.update_traces(marker=dict(size=20))

        fig.update_layout(
            mapbox_style="streets",
            mapbox_center={"lat": center_lat, "lon": center_lon},
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )

        return fig

    except Exception as e:
        return leere_karte(f"Fehler bei der Clusteranalyse: {str(e)}")


# ----------------------------
# Ausführung
# ----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)

    