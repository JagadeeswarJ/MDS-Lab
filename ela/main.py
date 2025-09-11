import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def read_data(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # try different date columns
    if 'sampling_date' in df.columns:
        df['sampling_date'] = pd.to_datetime(df['sampling_date'], errors='coerce')
    if 'date' in df.columns and df['date'].dtype == object:
        # if date exists and sampling_date was empty
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # create unified date column
    if 'sampling_date' in df.columns:
        df['dt'] = df['sampling_date']
    elif 'date' in df.columns:
        df['dt'] = df['date']
    else:
        raise ValueError('No date-like column found (sampling_date or date required)')
    return df


def clean_and_engineer(df):
    # basic cleaning
    df = df.copy()
    # drop rows with no date
    df = df.dropna(subset=['dt'])
    # numeric pollutants
    for col in ['so2','no2','rspm','spm','pm2_5']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # add year,month
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    df['ym'] = df['dt'].dt.to_period('M').dt.to_timestamp()
    # create pm10 proxy: prefer spm -> rspm
    if 'pm10' not in df.columns:
        if 'rspm' in df.columns:
            df['pm10'] = df['rspm']
        elif 'spm' in df.columns:
            df['pm10'] = df['spm']
        else:
            df['pm10'] = np.nan
    # standard city column
    if 'location' in df.columns:
        df['city'] = df['location'].str.strip()
    elif 'state' in df.columns:
        df['city'] = df['state'].str.strip()
    else:
        df['city'] = 'Unknown'
    return df


# ------------------------------ EDA ------------------------------

def basic_stats(df, outdir):
    stats = df[['so2','no2','pm2_5','pm10']].describe()
    stats.to_csv(os.path.join(outdir,'basic_stats.csv'))
    return stats


def plot_distributions(df, outdir):
    ensure_dir(outdir)
    pollutants = ['pm2_5','pm10','no2']
    for p in pollutants:
        if p not in df.columns:
            continue
        plt.figure(figsize=(8,4))
        sns.histplot(df[p].dropna(), kde=True)
        plt.title(f'Distribution of {p}')
        plt.xlabel(p)
        plt.tight_layout()
        fn = os.path.join(outdir,f'distribution_{p}.png')
        plt.savefig(fn)
        plt.close()


def correlation_heatmap(df, outdir):
    cols = ['so2','no2','pm2_5','pm10']
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    plt.figure(figsize=(6,4))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Pollutant Correlation')
    plt.tight_layout()
    fn = os.path.join(outdir,'correlation_heatmap.png')
    plt.savefig(fn)
    plt.close()
    return corr


def top_cities_table(df, outdir, topn=10):
    agg = df.groupby('city').agg({'pm2_5':'mean','pm10':'mean','no2':'mean'}).reset_index()
    agg = agg.sort_values('pm2_5', ascending=False).head(topn)
    agg.to_csv(os.path.join(outdir,f'top_{topn}_cities.csv'), index=False)
    # bar plot
    plt.figure(figsize=(10,5))
    sns.barplot(data=agg, x='pm2_5', y='city')
    plt.title('Top cities by average PM2.5')
    plt.xlabel('PM2.5 (mean)')
    plt.tight_layout()
    fn = os.path.join(outdir,'top_cities_pm2_5.png')
    plt.savefig(fn)
    plt.close()
    return agg



def monthly_timeseries(df, pollutant='pm2_5'):
    ts = df.groupby('ym').agg({pollutant:'mean'}).rename(columns={pollutant:'value'})
    ts = ts.asfreq('MS')
    return ts


def plot_timeseries(ts, outdir, name='pm2_5_monthly'):
    plt.figure(figsize=(10,4))
    ts['value'].plot()
    plt.title(name)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.tight_layout()
    fn = os.path.join(outdir,f'{name}.png')
    plt.savefig(fn)
    plt.close()


def decompose_and_plot(ts, outdir, model='additive'):
    ts_clean = ts['value'].interpolate()
    res = seasonal_decompose(ts_clean, model=model, period=12, extrapolate_trend='freq')
    fig = res.plot()
    fig.set_size_inches(10,8)
    fn = os.path.join(outdir,'decomposition.png')
    fig.savefig(fn)
    plt.close()
    return res


def arima_forecast(ts, periods=6):
    ts_clean = ts['value'].fillna(method='ffill').dropna()
    # train/test split
    train = ts_clean[:-periods]
    test = ts_clean[-periods:]

    best_aic = np.inf
    best_order = None
    best_model = None
    # small grid to keep runtime small
    for p in range(0,3):
        for d in range(0,2):
            for q in range(0,3):
                try:
                    model = ARIMA(train, order=(p,d,q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p,d,q)
                        best_model = model
                except Exception:
                    continue
    if best_model is None:
        raise RuntimeError('ARIMA modeling failed')
    # forecast
    fc_res = best_model.get_forecast(steps=periods)
    fc = fc_res.predicted_mean
    mae = mean_absolute_error(test, fc)
    return {'model':best_model, 'order':best_order, 'forecast':fc, 'mae':mae, 'test':test}



def spatial_plot_top_cities(df, outdir, topn=50):
    if 'latitude' in df.columns and 'longitude' in df.columns:
        agg = df.groupby('city').agg({'latitude':'mean','longitude':'mean','pm2_5':'mean'}).reset_index()
        agg = agg.dropna(subset=['latitude','longitude'])
        fig = px.scatter_mapbox(agg, lat='latitude', lon='longitude', size='pm2_5', hover_name='city', zoom=4,
                                title='PM2.5 by station (avg)')
        fig.update_layout(mapbox_style='open-street-map')
        fn = os.path.join(outdir,'pm2_5_map.html')
        fig.write_html(fn)
        return fn
    else:
        return None



def generate_report(outdir, stats, corr, top_cities, ts_summary, arima_res, notes):
    rpt = []
    rpt.append('# Problem Statement')
    rpt.append('Air pollution in Indian cities poses a significant risk to public health. This project analyzes PM2.5, PM10 (proxied by RSPM/SPM), and NO2 levels across urban monitoring stations to discover temporal and spatial patterns over the available timeframe (typically up to 5 years in the source dataset).')

    rpt.append('\n# Objective')
    rpt.append('1. Explore temporal trends and seasonality for PM2.5, PM10, and NO2.\n2. Identify cities and states with consistently high pollutant concentrations.\n3. Build a simple forecasting model to predict short-term PM2.5 levels.\n4. Produce visualizations and a summary report.')

    rpt.append('\n# Dataset Description')
    rpt.append('Columns present: stn_code, sampling_date, state, location, agency, type, so2, no2, rspm, spm, location_monitoring_station, pm2_5, date.\nNotes: `rspm` used as PM10 proxy when PM10 is absent. Dates parsed and aggregated monthly for time-series analysis.')

    rpt.append('\n# Exploratory Data Analysis (EDA)')
    rpt.append('Summary statistics:')
    rpt.append(stats.to_markdown())
    rpt.append('\nCorrelation matrix:')
    rpt.append(corr.to_markdown())
    rpt.append('\nTop cities by average PM2.5:')
    rpt.append(top_cities.to_markdown(index=False))
    rpt.append('\nKey plots saved in outputs/: distribution_*.png, correlation_heatmap.png, top_cities_pm2_5.png, and timeseries/decomposition images.')

    rpt.append('\n# Model Building / Insights')
    rpt.append(f'ARIMA model order chosen: {arima_res["order"]} with MAE on last {len(arima_res["test"])} months = {arima_res["mae"]:.2f}')
    rpt.append('Interpretation: Short-term forecasts are indicative but limited by data quality, missingness and the simple ARIMA model used. Seasonal patterns (monthly) were observed via decomposition.')

    rpt.append('\n# Conclusion')
    rpt.append('The analysis highlights seasonal and city-level disparities in PM2.5 and NO2. Top-polluted cities can be targeted for mitigation policies; short-term forecasting can inform alerts but requires more robust models and covariates like meteorology or emissions for operational usage.')

    rpt.append('\n# Future Scope')
    rpt.append('- Integrate meteorological features (wind, temperature, humidity) and mobility/activity data.\n- Use advanced forecasting models (SARIMAX with exogenous variables, Prophet, LSTM).\n- Spatial interpolation and high-resolution maps using geocoded station locations.\n- Build interactive dashboards (Dash/Streamlit) for live monitoring.')

    rpt.append('\n# References')
    rpt.append('- CPCB / OpenAQ India APIs for raw data.\n- statsmodels, pandas, plotly documentation for methods used.')

    rpt.append('\n# Notes & Reproducibility')
    rpt.append(notes)

    with open(os.path.join(outdir,'report.md'),'w',encoding='utf-8') as f:
        f.write('\n\n'.join(rpt))
    print(f'Report written to {os.path.join(outdir,"report.md")}')



def main(args):
    outdir = args.outdir
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir,'plots'))

    df = read_data(args.data)
    df = clean_and_engineer(df)

    # Basic stats and EDA
    stats = basic_stats(df, outdir)
    plot_distributions(df, os.path.join(outdir,'plots'))
    corr = correlation_heatmap(df, os.path.join(outdir,'plots'))
    top_cities = top_cities_table(df, os.path.join(outdir,'plots'))

    ts_pm25 = monthly_timeseries(df, pollutant='pm2_5')
    plot_timeseries(ts_pm25, os.path.join(outdir,'plots'), name='pm2_5_monthly')
    decomp = decompose_and_plot(ts_pm25, os.path.join(outdir,'plots'))

    try:
        arima_res = arima_forecast(ts_pm25, periods=6)
    except Exception as e:
        arima_res = {'order':None,'mae':np.nan,'forecast':None,'test':pd.Series(),'model':None}

    # Spatial
    spatial_fn = spatial_plot_top_cities(df, outdir)

    notes = 'Generated by automated script. Check outputs/plots for PNGs and pm2_5_map.html (if lat/lon available).'
    generate_report(outdir, stats, corr, top_cities, ts_pm25.describe(), arima_res, notes)

    print('\nDone. Files saved in', outdir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='CSV file path')
    p.add_argument('--outdir', default='outputs', help='Output directory')
    p.add_argument('--city', default=None, help='Optional: focus on single city')
    args = p.parse_args()
    main(args)
