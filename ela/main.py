import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def read_data(path):
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='cp1252')
    
    df.columns = [c.strip() for c in df.columns]
    
    # Handle date columns
    if 'sampling_date' in df.columns:
        df['date'] = pd.to_datetime(df['sampling_date'], errors='coerce')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Convert pollutant columns to numeric
    for col in ['so2', 'no2', 'rspm', 'spm', 'pm2_5']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create pm10 from rspm or spm if available
    if 'pm10' not in df.columns:
        if 'rspm' in df.columns:
            df['pm10'] = df['rspm']
        elif 'spm' in df.columns:
            df['pm10'] = df['spm']
    
    return df

def analyze_data(df, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    # Basic statistics
    pollutants = ['pm2_5', 'pm10', 'no2', 'so2']
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    print("Basic Statistics:")
    stats = df[available_pollutants].describe()
    print(stats)
    stats.to_csv(f"{outdir}/basic_stats.csv")
    
    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, pol in enumerate(available_pollutants[:4]):
        if pol in df.columns:
            df[pol].hist(bins=30, ax=axes[i])
            axes[i].set_title(f'{pol} Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/distributions.png")
    plt.close()
    
    # Correlation heatmap
    if len(available_pollutants) > 1:
        corr = df[available_pollutants].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Pollutant Correlations')
        plt.tight_layout()
        plt.savefig(f"{outdir}/correlation.png")
        plt.close()
    
    # Top cities by PM2.5 (if location data available)
    if 'location' in df.columns and 'pm2_5' in df.columns:
        top_cities = df.groupby('location')['pm2_5'].mean().nlargest(10)
        plt.figure(figsize=(10, 6))
        top_cities.plot(kind='bar')
        plt.title('Top 10 Cities by Average PM2.5')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{outdir}/top_cities.png")
        plt.close()
        
        print("\nTop 10 Cities by PM2.5:")
        print(top_cities)
    
    print(f"\nAnalysis complete. Results saved in {outdir}/")

def main():

    input_data = "data.csv"
    df = read_data(input_data)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    outdir = "outputs"
    analyze_data(df, outdir)

if __name__ == '__main__':
    main()