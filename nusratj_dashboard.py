# Importing all required libraries for data processing, visualization, and running the Dash web app
import pandas as pd
import numpy as np
import re
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# SECTION 1: DATA LOADING AND CLEANING
# ─────────────────────────────────────────────

# Loading all three raw datasets from the local data folder
cheese  = pd.read_csv('nusratjdata/cheese_data.csv')
weather = pd.read_csv('nusratjdata/canada_weather.csv')
temp    = pd.read_csv('nusratjdata/Canada_Temperature_Data.csv')

# Defining the ten valid Canadian provinces
valid_provinces = ['QC', 'ON', 'BC', 'AB', 'MB', 'SK', 'NS', 'NB', 'NL', 'PE']

# Defining full province names for labels and tooltips
prov_names = {
    'QC': 'Quebec', 'ON': 'Ontario', 'BC': 'British Columbia',
    'AB': 'Alberta', 'MB': 'Manitoba', 'SK': 'Saskatchewan',
    'NS': 'Nova Scotia', 'NB': 'New Brunswick',
    'NL': 'Newfoundland', 'PE': 'Prince Edward Island'
}

# Cleaning the cheese dataset
cheese_clean = cheese.dropna(subset=['ManufacturerProvCode']).copy()
cheese_clean.rename(columns={'ManufacturerProvCode': 'Province'}, inplace=True)
cheese_clean = cheese_clean[cheese_clean['Province'].isin(valid_provinces)]
for col in ['CategoryTypeEn', 'MilkTypeEn', 'MilkTreatmentTypeEn', 'RindTypeEn', 'FatLevel']:
    cheese_clean[col] = cheese_clean[col].fillna('Unknown')
cheese_clean['ProvName'] = cheese_clean['Province'].map(prov_names)

# Defining a function that extracts Celsius values from mixed text strings
def extract_celsius(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('\u2212', '-').replace('\u2013', '-')
    match = re.match(r'^\s*([-]?\d+\.?\d*)', val)
    return float(match.group(1)) if match else np.nan

# Cleaning the weather dataset and averaging by province
weather['Province']     = weather['Community'].str.extract(r',\s*([A-Z]{2})$')
weather['AnnualHigh_C'] = weather['Annual(Avg. high °C (°F))'].apply(extract_celsius)
weather['AnnualLow_C']  = weather['Annual(Avg. low °C (°F))'].apply(extract_celsius)
weather['JanHigh_C']    = weather['January(Avg. high °C (°F))'].apply(extract_celsius)
weather['JulyHigh_C']   = weather['July(Avg. high °C (°F))'].apply(extract_celsius)
prov_weather = weather.groupby('Province').agg(
    AvgAnnualHigh=('AnnualHigh_C', 'mean'),
    AvgAnnualLow=('AnnualLow_C',  'mean'),
    AvgJanHigh=('JanHigh_C',      'mean'),
    AvgJulyHigh=('JulyHigh_C',    'mean')
).reset_index()

# Cleaning the temperature dataset and computing long term provincial averages
temp_clean = temp[temp['Prov'].isin(valid_provinces)].copy().dropna(subset=['Tm'])
prov_temp  = temp_clean.groupby('Prov')['Tm'].mean().reset_index()
prov_temp.columns = ['Province', 'AvgMeanTemp_C']

# Computing cheese count and average moisture per province
cheese_count = cheese_clean.groupby('Province').size().reset_index(name='CheeseCount')
moisture     = cheese_clean.groupby('Province')['MoisturePercent'].mean().reset_index()
moisture.columns = ['Province', 'AvgMoisture']

# Merging all data into one master dataframe
merged = cheese_count \
    .merge(prov_temp, on='Province', how='left') \
    .merge(prov_weather[prov_weather['Province'].isin(valid_provinces)], on='Province', how='left') \
    .merge(moisture, on='Province', how='left')
merged['ProvName'] = merged['Province'].map(prov_names)

# Simplifying cheese category names
cat_map = {
    'Firm Cheese': 'Firm', 'Semi-soft Cheese': 'Semi-soft',
    'Soft Cheese': 'Soft', 'Hard Cheese': 'Hard',
    'Fresh Cheese': 'Fresh', 'Veined Cheeses': 'Veined'
}
cheese_clean['Category'] = cheese_clean['CategoryTypeEn'].map(cat_map).fillna('Other')

# Preparing grouped dataframes for charts
cat_counts  = cheese_clean.groupby(['Province', 'Category']).size().reset_index(name='Count')
cat_counts['ProvName'] = cat_counts['Province'].map(prov_names)
milk_counts = cheese_clean.groupby(['Province', 'MilkTypeEn']).size().reset_index(name='Count')
milk_counts['ProvName'] = milk_counts['Province'].map(prov_names)

# ─────────────────────────────────────────────
# SECTION 2: CREATIVE METRIC CALCULATIONS
# ─────────────────────────────────────────────

# Calculating the Cheese Richness Score by combining moisture, fat, diversity and organic ratio
# Mapping fat level text values to numeric scores so they can be included in the formula
fat_map = {'lower fat': 1, 'light': 2, 'medium': 3, 'high': 4, 'extra high': 5}
cheese_clean['FatScore'] = cheese_clean['FatLevel'].str.lower().map(fat_map).fillna(3)

# Aggregating all ingredients of the richness score per province
richness_raw = cheese_clean.groupby('Province').agg(
    AvgMoisture=('MoisturePercent', 'mean'),
    AvgFat=('FatScore', 'mean'),
    OrganicRatio=('Organic', 'mean'),
    UniqueMilkTypes=('MilkTypeEn', 'nunique')
).reset_index()

# Normalizing each component to a 0 to 1 scale before combining into the final score
def norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx != mn else s * 0

richness_raw['RichnessScore'] = (
    norm(richness_raw['AvgMoisture'])   * 25 +
    norm(richness_raw['AvgFat'])        * 25 +
    norm(richness_raw['UniqueMilkTypes']) * 30 +
    richness_raw['OrganicRatio']        * 20
)
richness_raw['ProvName'] = richness_raw['Province'].map(prov_names)
richness_raw = richness_raw.sort_values('RichnessScore', ascending=True)

# Calculating the proportion of months per year where mean temperature is below 5 degrees Celsius
# This measures how cold each province is and is used to compare with cheese production style
cold_ratio = (
    temp_clean[temp_clean['Tm'] < 5].groupby('Prov').size() /
    temp_clean.groupby('Prov').size()
).reset_index()
cold_ratio.columns = ['Province', 'ColdMonthRatio']
cold_ratio['ColdMonths'] = (cold_ratio['ColdMonthRatio'] * 12).round(1)
cold_ratio['ProvName']   = cold_ratio['Province'].map(prov_names)

# Merging cold months data with cheese count for the correlation scatter chart
cold_merged = cold_ratio.merge(cheese_count, on='Province', how='left')
cold_merged['ProvName'] = cold_merged['Province'].map(prov_names)
cold_merged = cold_merged.merge(prov_temp, on='Province', how='left')

# Calculating the Shannon Diversity Index for milk types per province
# This measures how evenly distributed different milk types are within each province
milk_valid = cheese_clean[cheese_clean['MilkTypeEn'] != 'Unknown'].copy()
milk_grouped = milk_valid.groupby(['Province', 'MilkTypeEn']).size().reset_index(name='Count')

def shannon_index(group):
    # Calculating the Shannon entropy which increases when more milk types are used equally
    total = group['Count'].sum()
    p = group['Count'] / total
    return -np.sum(p * np.log(p + 1e-12))

diversity = milk_grouped.groupby('Province').apply(shannon_index).reset_index()
diversity.columns = ['Province', 'ShannonIndex']
diversity['ProvName'] = diversity['Province'].map(prov_names)
diversity = diversity.merge(cheese_count, on='Province', how='left')
diversity = diversity.sort_values('ShannonIndex', ascending=True)

# ─────────────────────────────────────────────
# SECTION 3: DESIGN TOKENS
# ─────────────────────────────────────────────

PALETTE   = ['#00C9A7','#845EC2','#FF6F91','#FFC75F','#F9F871','#0081CF','#D65DB1','#FF9671','#4E9F3D','#C9B1FF']
BG_DARK   = '#0D1117'
BG_CARD   = '#161B22'
BG_CARD2  = '#1C2128'
BORDER    = '#30363D'
TEXT_MAIN = '#E6EDF3'
TEXT_DIM  = '#8B949E'
ACCENT    = '#00C9A7'
ACCENT2   = '#845EC2'
FONT      = 'DM Sans'

# ─────────────────────────────────────────────
# SECTION 4: CHART BUILDING FUNCTIONS
# ─────────────────────────────────────────────

# Defining a reusable theme function that takes an optional legend override to avoid conflicts
def apply_theme(fig, title, xlab='', ylab='', legend_override=None):
    legend_cfg = dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER, borderwidth=1, font=dict(size=11, color=TEXT_MAIN))
    if legend_override:
        legend_cfg.update(legend_override)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT, color=TEXT_MAIN, size=13),
        title=dict(text=title, font=dict(family=FONT, size=19, color=TEXT_MAIN)),
        legend=legend_cfg,
        xaxis=dict(
            title=xlab, showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            tickfont=dict(size=12, color=TEXT_MAIN),
            title_font=dict(size=13, color=TEXT_DIM),
            linecolor=BORDER
        ),
        yaxis=dict(
            title=ylab, showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            tickfont=dict(size=12, color=TEXT_MAIN),
            title_font=dict(size=13, color=TEXT_DIM),
            linecolor=BORDER
        ),
        margin=dict(l=60, r=40, t=70, b=60),
        hoverlabel=dict(
            bgcolor='#1C2128',
            bordercolor=ACCENT,
            font=dict(family=FONT, size=13, color=TEXT_MAIN),
            align='left',
            namelength=0
        ),
        transition=dict(duration=500, easing='cubic-in-out')
    )
    return fig


# Defining the bubble chart showing cheese count vs average temperature
def build_bubble(selected_provs):
    df  = merged[merged['Province'].isin(selected_provs)].copy() if selected_provs else merged.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['AvgMeanTemp_C'],
        y=df['CheeseCount'],
        mode='markers+text',
        text=df['Province'],
        textposition='top center',
        textfont=dict(size=11, color='white', family=FONT),
        marker=dict(
            size=[max(r['CheeseCount'] ** 0.52, 16) for _, r in df.iterrows()],
            color=df['AvgMeanTemp_C'],
            colorscale=[[0.0,'#845EC2'],[0.3,'#0081CF'],[0.6,'#00C9A7'],[1.0,'#FFC75F']],
            cmin=merged['AvgMeanTemp_C'].min(),
            cmax=merged['AvgMeanTemp_C'].max(),
            showscale=True,
            colorbar=dict(
                title=dict(text='Temp (°C)', font=dict(color=TEXT_DIM, size=11)),
                tickfont=dict(color=TEXT_DIM, size=10),
                thickness=10, len=0.6, x=1.02
            ),
            line=dict(color='rgba(255,255,255,0.25)', width=1.5),
            opacity=0.9
        ),
        customdata=np.stack([df['ProvName'], df['CheeseCount'], df['AvgMeanTemp_C'], df['AvgMoisture']], axis=1),
        hovertemplate=(
            '<b style="font-size:14px">%{customdata[0]}</b><br>'
            '\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015<br>'
            '\U0001f321\ufe0f  Avg Temp: <b>%{customdata[2]:.1f} \u00b0C</b><br>'
            '\U0001f9c0  Cheeses: <b>%{customdata[1]:.0f}</b><br>'
            '\U0001f4a7  Avg Moisture: <b>%{customdata[3]:.1f}%</b>'
            '<extra></extra>'
        ),
        name='Province'
    ))
    z  = np.polyfit(df['AvgMeanTemp_C'].fillna(0), df['CheeseCount'], 1)
    p  = np.poly1d(z)
    xs = np.linspace(merged['AvgMeanTemp_C'].min() - 0.5, merged['AvgMeanTemp_C'].max() + 0.5, 300)
    fig.add_trace(go.Scatter(
        x=xs, y=p(xs), mode='lines',
        line=dict(color='rgba(0,201,167,0.35)', width=2, dash='dot'),
        hoverinfo='skip', showlegend=False
    ))
    apply_theme(fig, 'Chart 1 \u2014 Cheese Production vs Average Temperature',
                xlab='Avg Historical Mean Temperature (\u00b0C)', ylab='Number of Cheeses')
    fig.update_layout(showlegend=False, hovermode='closest')
    return fig


# Defining the stacked horizontal bar chart showing cheese categories per province
def build_category_bar(selected_provs):
    df    = cat_counts[cat_counts['Province'].isin(selected_provs)].copy() if selected_provs else cat_counts.copy()
    cats  = sorted(df['Category'].unique().tolist())
    provs = df['Province'].unique().tolist()
    fig   = go.Figure()
    for i, cat in enumerate(cats):
        sub = df[df['Category'] == cat].set_index('Province')['Count']
        fig.add_trace(go.Bar(
            name=cat,
            y=[prov_names.get(p, p) for p in provs],
            x=[sub.get(p, 0) for p in provs],
            orientation='h',
            marker=dict(color=PALETTE[i % len(PALETTE)], line=dict(width=0), opacity=0.88),
            hovertemplate='<b>%{y}</b><br>' + cat + ': <b>%{x}</b><extra></extra>'
        ))
    apply_theme(fig, 'Chart 2 \u2014 Cheese Category Breakdown by Province',
                xlab='Number of Cheeses',
                legend_override=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_layout(barmode='stack', hovermode='y unified')
    return fig


# Defining the donut chart showing milk type distribution
def build_milk_donut(selected_provs):
    df      = milk_counts[milk_counts['Province'].isin(selected_provs)].copy() if selected_provs else milk_counts.copy()
    grouped = df[df['MilkTypeEn'] != 'Unknown'].groupby('MilkTypeEn')['Count'].sum().reset_index()
    grouped = grouped.sort_values('Count', ascending=False)
    fig = go.Figure(go.Pie(
        labels=grouped['MilkTypeEn'],
        values=grouped['Count'],
        hole=0.58,
        marker=dict(colors=PALETTE[:len(grouped)], line=dict(color=BG_DARK, width=3)),
        textfont=dict(family=FONT, size=12, color='white'),
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>Count: <b>%{value}</b><br>Share: <b>%{percent}</b><extra></extra>',
        pull=[0.04 if i == 0 else 0.01 for i in range(len(grouped))]
    ))
    apply_theme(fig, 'Chart 3 \u2014 Milk Type Distribution')
    fig.update_layout(
        annotations=[dict(text='Milk<br>Types', x=0.5, y=0.5, font=dict(size=15, family=FONT, color=TEXT_DIM), showarrow=False)],
        legend=dict(orientation='v', x=1.02, y=0.5, font=dict(size=12, color=TEXT_MAIN), bgcolor='rgba(0,0,0,0)')
    )
    return fig


# Defining the seasonal temperature grouped bar chart with crosshair hover effect
def build_temp_range(selected_provs):
    df  = merged[merged['Province'].isin(selected_provs)].copy() if selected_provs else merged.copy()
    df  = df.sort_values('AvgMeanTemp_C')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='January High', x=df['ProvName'], y=df['AvgJanHigh'],
        marker=dict(color='#0081CF', line=dict(width=0), opacity=0.88),
        hovertemplate='Jan High: <b>%{y:.1f} \u00b0C</b><extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='July High', x=df['ProvName'], y=df['AvgJulyHigh'],
        marker=dict(color='#FFC75F', line=dict(width=0), opacity=0.88),
        hovertemplate='July High: <b>%{y:.1f} \u00b0C</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df['ProvName'], y=df['CheeseCount'],
        name='Cheese Count', mode='lines+markers', yaxis='y2',
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=9, color=ACCENT, line=dict(color='white', width=2)),
        hovertemplate='Cheeses: <b>%{y}</b><extra></extra>'
    ))
    apply_theme(fig, 'Chart 4 \u2014 Seasonal Temperature Range vs Cheese Count',
                ylab='Temperature (\u00b0C)',
                legend_override=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_layout(
        barmode='group', hovermode='x unified',
        xaxis=dict(showspikes=True, spikecolor=ACCENT, spikethickness=1.5, spikedash='dot', spikemode='across', spikesnap='cursor'),
        yaxis2=dict(title='Number of Cheeses', overlaying='y', side='right', showgrid=False,
                    tickfont=dict(color=ACCENT, size=12), title_font=dict(color=ACCENT, size=13))
    )
    return fig


# Defining the heatmap showing cheese category intensity across provinces ordered by temperature
def build_heatmap(selected_provs):
    df = cat_counts[cat_counts['Province'].isin(selected_provs)].copy() if selected_provs else cat_counts.copy()

    # Ordering provinces from coldest to warmest so the temperature gradient reads left to right
    temp_order = merged[merged['Province'].isin(selected_provs if selected_provs else valid_provinces)] \
        .sort_values('AvgMeanTemp_C')['Province'].tolist()

    cats  = sorted(df['Category'].unique().tolist())
    pivot = df.groupby(['Province', 'Category'])['Count'].sum().unstack(fill_value=0)
    pivot = pivot.reindex(temp_order).fillna(0)

    z      = [pivot[cat].tolist() if cat in pivot.columns else [0]*len(temp_order) for cat in cats]
    x_lbls = [prov_names.get(p, p) for p in temp_order]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_lbls,
        y=cats,
        colorscale=[
            [0.0,  '#0D1117'],
            [0.05, '#1B2A3B'],
            [0.2,  '#0081CF'],
            [0.5,  '#845EC2'],
            [0.8,  '#FF6F91'],
            [1.0,  '#FFC75F']
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text='Count', font=dict(color=TEXT_DIM, size=11)),
            tickfont=dict(color=TEXT_DIM, size=10),
            thickness=12
        ),
        hovertemplate='<b>%{x}</b><br>Category: <b>%{y}</b><br>Count: <b>%{z}</b><extra></extra>',
        xgap=3, ygap=3
    ))

    apply_theme(fig, 'Chart 5 \u2014 Cheese Style Heatmap \u2014 Cold Provinces \u2192 Warm Provinces',
                xlab='Provinces ordered coldest \u2192 warmest', ylab='Cheese Category')
    fig.update_layout(hovermode='closest')
    return fig


# Defining the Cheese Richness Score horizontal bar chart with gradient fill
def build_richness(selected_provs):
    df = richness_raw[richness_raw['Province'].isin(selected_provs)].copy() if selected_provs else richness_raw.copy()
    df = df.sort_values('RichnessScore', ascending=True)

    # Coloring bars by score so higher richness provinces appear in warmer colors
    colors = [f'rgba(0,{int(201*(s/100))},{int(167*(s/100))},0.88)' for s in df['RichnessScore']]

    fig = go.Figure(go.Bar(
        x=df['RichnessScore'],
        y=df['ProvName'],
        orientation='h',
        marker=dict(
            color=df['RichnessScore'],
            colorscale=[[0,'#845EC2'],[0.4,'#0081CF'],[0.7,'#00C9A7'],[1,'#FFC75F']],
            line=dict(width=0),
            opacity=0.92
        ),
        customdata=np.stack([
            df['AvgMoisture'], df['AvgFat'],
            df['OrganicRatio']*100, df['UniqueMilkTypes']
        ], axis=1),
        hovertemplate=(
            '<b>%{y}</b><br>'
            '\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015<br>'
            'Richness Score: <b>%{x:.1f} / 100</b><br>'
            '\U0001f4a7 Avg Moisture: <b>%{customdata[0]:.1f}%</b><br>'
            '\U0001f9c0 Avg Fat Score: <b>%{customdata[1]:.1f}</b><br>'
            '\U0001f33f Organic Ratio: <b>%{customdata[2]:.1f}%</b><br>'
            '\U0001f404 Unique Milk Types: <b>%{customdata[3]:.0f}</b>'
            '<extra></extra>'
        )
    ))

    # Adding a vertical reference line at the average richness score
    avg_score = df['RichnessScore'].mean()
    fig.add_vline(
        x=avg_score,
        line=dict(color='rgba(255,255,255,0.2)', width=1.5, dash='dot'),
        annotation_text=f'Avg: {avg_score:.1f}',
        annotation_font=dict(color=TEXT_DIM, size=11)
    )

    apply_theme(fig, 'Chart 6 \u2014 Cheese Richness Score by Province',
                xlab='Richness Score (0\u2013100)', ylab='')
    fig.update_layout(hovermode='closest', showlegend=False)
    return fig


# Defining the cold months vs cheese count scatter chart showing the temperature-production relationship
def build_cold_months(selected_provs):
    df = cold_merged[cold_merged['Province'].isin(selected_provs)].copy() if selected_provs else cold_merged.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['ColdMonths'],
        y=df['CheeseCount'],
        mode='markers+text',
        text=df['Province'],
        textposition='top center',
        textfont=dict(size=11, color='white', family=FONT),
        marker=dict(
            size=14,
            color=df['AvgMeanTemp_C'],
            colorscale=[[0.0,'#845EC2'],[0.5,'#0081CF'],[1.0,'#00C9A7']],
            line=dict(color='rgba(255,255,255,0.3)', width=1.5),
            opacity=0.9,
            showscale=True,
            colorbar=dict(
                title=dict(text='Avg Temp (°C)', font=dict(color=TEXT_DIM, size=11)),
                tickfont=dict(color=TEXT_DIM, size=10),
                thickness=10, len=0.5, x=1.02
            )
        ),
        customdata=np.stack([df['ProvName'], df['CheeseCount'], df['ColdMonths'], df['AvgMeanTemp_C']], axis=1),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            '\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015<br>'
            '\u2744\ufe0f  Cold Months per Year: <b>%{customdata[2]:.1f}</b><br>'
            '\U0001f9c0  Cheeses Produced: <b>%{customdata[1]:.0f}</b><br>'
            '\U0001f321\ufe0f  Avg Temp: <b>%{customdata[3]:.1f} \u00b0C</b>'
            '<extra></extra>'
        ),
        name=''
    ))

    # Adding a trend line to show the overall direction of the cold months vs cheese count relationship
    z  = np.polyfit(df['ColdMonths'], df['CheeseCount'], 1)
    p  = np.poly1d(z)
    xs = np.linspace(df['ColdMonths'].min() - 0.2, df['ColdMonths'].max() + 0.2, 200)
    fig.add_trace(go.Scatter(
        x=xs, y=p(xs), mode='lines',
        line=dict(color='rgba(132,94,194,0.4)', width=2, dash='dot'),
        hoverinfo='skip', showlegend=False
    ))

    apply_theme(fig, 'Chart 7 \u2014 Cold Months per Year vs Cheese Production',
                xlab='Estimated Cold Months per Year (Avg Temp < 5\u00b0C)',
                ylab='Number of Cheeses Produced')
    fig.update_layout(showlegend=False, hovermode='closest')
    return fig


# Defining the Shannon Diversity Index bar chart showing milk type diversity per province
def build_diversity(selected_provs):
    df = diversity[diversity['Province'].isin(selected_provs)].copy() if selected_provs else diversity.copy()
    df = df.sort_values('ShannonIndex', ascending=True)

    fig = go.Figure(go.Bar(
        x=df['ShannonIndex'],
        y=df['ProvName'],
        orientation='h',
        marker=dict(
            color=df['ShannonIndex'],
            colorscale=[[0,'#1C2128'],[0.3,'#845EC2'],[0.7,'#0081CF'],[1,'#00C9A7']],
            line=dict(width=0),
            opacity=0.9
        ),
        customdata=np.stack([df['ProvName'], df['ShannonIndex'], df['CheeseCount']], axis=1),
        hovertemplate=(
            '<b>%{y}</b><br>'
            '\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015<br>'
            'Shannon Diversity Index: <b>%{x:.3f}</b><br>'
            '\U0001f404 Higher = more balanced milk type variety<br>'
            '\U0001f9c0 Total Cheeses: <b>%{customdata[2]:.0f}</b>'
            '<extra></extra>'
        )
    ))

    # Adding an average reference line so viewers can see which provinces are above or below average diversity
    avg = df['ShannonIndex'].mean()
    fig.add_vline(
        x=avg,
        line=dict(color='rgba(255,255,255,0.2)', width=1.5, dash='dot'),
        annotation_text=f'Avg: {avg:.3f}',
        annotation_font=dict(color=TEXT_DIM, size=11)
    )

    apply_theme(fig, 'Chart 8 \u2014 Milk Type Diversity Index by Province (Shannon Index)',
                xlab='Shannon Diversity Index (higher = more diverse milk types)', ylab='')
    fig.update_layout(hovermode='closest', showlegend=False)
    return fig


# ─────────────────────────────────────────────
# SECTION 5: DASH APP LAYOUT
# ─────────────────────────────────────────────

# Initializing the Dash app with Google DM Sans font loaded from Google Fonts
app = dash.Dash(
    __name__,
    title='Canadian Cheese x Climate Dashboard',
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap'
    ]
)

# Defining reusable style dictionaries
CARD = dict(background=BG_CARD, border=f'1px solid {BORDER}', borderRadius='16px', padding='28px', marginBottom='24px')
LBL  = dict(fontFamily=FONT, color=TEXT_DIM, fontSize='11px', fontWeight='600', letterSpacing='1.5px', textTransform='uppercase', marginBottom='10px', display='block')

# Defining a helper that creates a section divider label between chart groups
def section_label(text):
    return html.Div(style=dict(marginBottom='16px', marginTop='8px'), children=[
        html.Span(text, style=dict(
            fontFamily=FONT, color=ACCENT, fontSize='11px',
            fontWeight='700', letterSpacing='2px', textTransform='uppercase',
            borderLeft=f'3px solid {ACCENT}', paddingLeft='10px'
        ))
    ])

# Building the complete page layout with header, KPI cards, filter, and 8 full width charts
app.layout = html.Div(
    style=dict(background=BG_DARK, minHeight='100vh', fontFamily=FONT, padding='32px 48px', maxWidth='1400px', margin='0 auto'),
    children=[

        # Page header with gradient accent bar
        html.Div(style=dict(marginBottom='36px'), children=[
            html.Div(style=dict(display='flex', alignItems='center', gap='14px', marginBottom='8px'), children=[
                html.Div(style=dict(width='6px', height='46px', background=f'linear-gradient(180deg, {ACCENT}, {ACCENT2})', borderRadius='3px')),
                html.H1('Canadian Cheese \u00d7 Climate', style=dict(fontFamily=FONT, color=TEXT_MAIN, fontSize='34px', fontWeight='700', margin='0', letterSpacing='-0.5px'))
            ]),
            html.P('Exploring the relationship between provincial temperature patterns and cheese production in Canada',
                   style=dict(fontFamily=FONT, color=TEXT_DIM, fontSize='15px', margin='0 0 0 20px')),
            html.P('Nusrat Jahan  \u00b7  Data Analyst Assessment  \u00b7  Canadian Sheep Federation 2026',
                   style=dict(fontFamily=FONT, color=ACCENT, fontSize='12px', fontWeight='500', margin='6px 0 0 20px', letterSpacing='0.5px'))
        ]),

        # Four KPI summary cards
        html.Div(style=dict(display='grid', gridTemplateColumns='repeat(4, 1fr)', gap='16px', marginBottom='28px'), children=[
            html.Div(style=CARD, children=[
                html.Span('Total Cheeses', style=LBL),
                html.H2(f"{cheese_clean.shape[0]:,}", style=dict(fontFamily=FONT, color=ACCENT, fontSize='38px', fontWeight='700', margin='0')),
                html.P('across all provinces', style=dict(color=TEXT_DIM, fontSize='12px', margin='4px 0 0'))
            ]),
            html.Div(style=CARD, children=[
                html.Span('Provinces Analyzed', style=LBL),
                html.H2('10', style=dict(fontFamily=FONT, color=ACCENT2, fontSize='38px', fontWeight='700', margin='0')),
                html.P('Canadian provinces', style=dict(color=TEXT_DIM, fontSize='12px', margin='4px 0 0'))
            ]),
            html.Div(style=CARD, children=[
                html.Span('Top Producer', style=LBL),
                html.H2('Quebec', style=dict(fontFamily=FONT, color='#FF6F91', fontSize='30px', fontWeight='700', margin='0')),
                html.P(f"{int(merged[merged['Province']=='QC']['CheeseCount'].values[0]):,} cheeses (76%)", style=dict(color=TEXT_DIM, fontSize='12px', margin='4px 0 0'))
            ]),
            html.Div(style=CARD, children=[
                html.Span('Coldest High Producer', style=LBL),
                html.H2('Quebec', style=dict(fontFamily=FONT, color='#0081CF', fontSize='30px', fontWeight='700', margin='0')),
                html.P(f"Avg {merged[merged['Province']=='QC']['AvgMeanTemp_C'].values[0]:.1f} \u00b0C mean temp", style=dict(color=TEXT_DIM, fontSize='12px', margin='4px 0 0'))
            ]),
        ]),

        # Province filter dropdown
        html.Div(style={**CARD, 'marginBottom': '28px'}, children=[
            html.Span('Filter by Province', style=LBL),
            dcc.Dropdown(
                id='prov-filter',
                options=[{'label': prov_names[p], 'value': p} for p in valid_provinces],
                value=valid_provinces,
                multi=True,
                style=dict(fontFamily=FONT, backgroundColor=BG_CARD2, color=TEXT_MAIN, border=f'1px solid {BORDER}', borderRadius='8px')
            )
        ]),

        # Section A: Core analysis charts
        section_label('Core Analysis'),

        html.Div(style=CARD, children=[dcc.Graph(id='bubble-chart',   config={'displayModeBar': False}, style=dict(height='500px'))]),
        html.Div(style=CARD, children=[dcc.Graph(id='category-bar',   config={'displayModeBar': False}, style=dict(height='520px'))]),
        html.Div(style=CARD, children=[dcc.Graph(id='milk-donut',     config={'displayModeBar': False}, style=dict(height='480px'))]),
        html.Div(style=CARD, children=[dcc.Graph(id='temp-range',     config={'displayModeBar': False}, style=dict(height='480px'))]),

        # Section B: Creative analysis charts
        section_label('Creative Analysis \u2014 Beyond the Basics'),

        html.Div(style=CARD, children=[dcc.Graph(id='heatmap-chart',  config={'displayModeBar': False}, style=dict(height='420px'))]),
        html.Div(style=CARD, children=[dcc.Graph(id='richness-chart', config={'displayModeBar': False}, style=dict(height='460px'))]),
        html.Div(style=CARD, children=[dcc.Graph(id='cold-chart',     config={'displayModeBar': False}, style=dict(height='500px'))]),
        html.Div(style=CARD, children=[dcc.Graph(id='diversity-chart',config={'displayModeBar': False}, style=dict(height='460px'))]),

        # Footer
        html.Div(style=dict(textAlign='center', paddingTop='24px', borderTop=f'1px solid {BORDER}', marginTop='8px'), children=[
            html.P('Submitted by Nusrat Jahan  \u00b7  Data Analyst Internship  \u00b7  Canadian Sheep Federation 2026',
                   style=dict(color=TEXT_DIM, fontFamily=FONT, fontSize='12px', margin='0'))
        ])
    ]
)

# ─────────────────────────────────────────────
# SECTION 6: CALLBACKS
# ─────────────────────────────────────────────

# Defining the callback that updates all eight charts whenever the province filter changes
@app.callback(
    Output('bubble-chart',   'figure'),
    Output('category-bar',   'figure'),
    Output('milk-donut',     'figure'),
    Output('temp-range',     'figure'),
    Output('heatmap-chart',  'figure'),
    Output('richness-chart', 'figure'),
    Output('cold-chart',     'figure'),
    Output('diversity-chart','figure'),
    Input('prov-filter',     'value')
)
def update_all_charts(selected_provs):
    # Returning all eight updated chart figures based on the currently selected provinces
    return (
        build_bubble(selected_provs),
        build_category_bar(selected_provs),
        build_milk_donut(selected_provs),
        build_temp_range(selected_provs),
        build_heatmap(selected_provs),
        build_richness(selected_provs),
        build_cold_months(selected_provs),
        build_diversity(selected_provs)
    )


# ─────────────────────────────────────────────
# SECTION 7: RUN
# ─────────────────────────────────────────────

# Starting the Dash server on port 8050
if __name__ == '__main__':
    app.run(debug=False, port=8050)