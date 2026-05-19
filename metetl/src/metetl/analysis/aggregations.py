import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import logging

logger = logging.getLogger("metetl.analysis")

def file_reader(path, chunk_size=5000):
    cols = [
        'AccessionYear', 'Object Begin Date', 'Department', 
        'Culture', 'Medium', 'Classification', 'Country'
    ]
    logger.debug(f"Reading CSV: {path} (chunk size: {chunk_size})")
    return pd.read_csv(path, chunksize=chunk_size, low_memory=False, usecols=cols)

def data_cleaner(chunk_df):
    chunk_df['AccessionYear'] = pd.to_numeric(chunk_df['AccessionYear'], errors='coerce')
    df = chunk_df.dropna(subset=['AccessionYear', 'Object Begin Date']).copy()
    df['Age'] = df['AccessionYear'] - df['Object Begin Date']
    df['Decade'] = (df['AccessionYear'] // 10) * 10
    return df[df['Age'] >= 0]

def calculate_metrics(df):
    results = pd.DataFrame()
    for field in df.columns:
        counts = df[field].values
        counts = counts[counts > 0]
        if len(counts) == 0: continue
        
        p = counts / counts.sum()
        n = len(p)
        
        gini = 1 - np.sum(p ** 2)
        entropy = -np.sum(p * np.log2(p + 1e-12))
        max_entropy = np.log2(n) if n > 1 else 1
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0
        enc_norm = (1 / np.sum(p ** 2)) / n if n > 0 else 0
        
        temp = pd.DataFrame({
            'Gini': [gini], 
            'Entropy_norm': [entropy_norm], 
            'ENC_norm': [enc_norm]
        }, index=[field])
        results = pd.concat([results, temp])
    return results

def run_analysis(csv_path, output_dir):
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting statistical analysis of the dataset")

    stat_summary = pd.DataFrame(columns=['size', 'sum', 'sum_sq'])
    total_counts = pd.DataFrame()
    col_freq = ['Department', 'Culture', 'Medium', 'Classification', 'Country']

    for raw_chunk in file_reader(csv_path):
        chunk_df = data_cleaner(raw_chunk)
        if chunk_df.empty: continue

        chunk_info = pd.DataFrame({
            'size': chunk_df.groupby('Decade')['Age'].size(),
            'sum': chunk_df.groupby('Decade')['Age'].sum(),
            'sum_sq': (chunk_df['Age']**2).groupby(chunk_df['Decade']).sum()
        })
        stat_summary = stat_summary.add(chunk_info, fill_value=0)

        for field in col_freq:
            counts = chunk_df[field].value_counts().reset_index()
            counts.columns = ['value', 'count']
            counts['field'] = field
            total_counts = pd.concat([total_counts, counts], ignore_index=True)

    if stat_summary.empty:
        logger.warning("No data available for analysis.")
        return

    size = stat_summary['size'].astype(float)
    sum_val = stat_summary['sum'].astype(float)
    sum_sq = stat_summary['sum_sq'].astype(float)

    stat_summary['mean_age'] = sum_val / size
    variance = (sum_sq - (sum_val**2 / size)) / (size - 1)
    stat_summary['std_age'] = np.sqrt(variance.clip(lower=0))
    stat_summary = stat_summary.reset_index().sort_values('Decade')
    
    stat_summary['ci_95'] = 1.96 * stat_summary['std_age'] / np.sqrt(size)
    stat_summary['scatter_limit'] = 1.96 * stat_summary['std_age']             
    stat_summary['age_diff'] = stat_summary['mean_age'].diff()

    plot_age_statistics(stat_summary, output_dir)
    
    logger.info("Calculating quality metrics")
    freq_result = total_counts.groupby(['field', 'value'])['count'].sum().unstack(level=0).fillna(0)
    metrics = calculate_metrics(freq_result)
    plot_quality_heatmap(metrics, output_dir)

    logger.info(f"Analysis finished successfully. Results saved in: {output_dir}")

def plot_age_statistics(stat, output_dir):
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    decades_str = stat['Decade'].astype(int).astype(str)
    low_scatter = np.minimum(stat['mean_age'], stat['scatter_limit'])
    
    ax[0].bar(decades_str, stat['mean_age'], color='skyblue', label='Mean Age')
    ax[0].errorbar(decades_str, stat['mean_age'], 
                   yerr=[low_scatter, stat['scatter_limit']], 
                   color='lightgray', fmt='none', label='Std Deviation')
    ax[0].errorbar(decades_str, stat['mean_age'], 
                   yerr=stat['ci_95'], 
                   color='black', capsize=5, fmt='none', label='95% CI')
    
    ax[0].set_title('Average Object Age by Accession Decade')
    ax[0].set_ylabel('Age (years)')
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.3)

    ax[1].plot(decades_str, stat['age_diff'], marker='o', color='red', label='Age Diff')
    ax[1].axhline(0, color='black', linewidth=1)
    ax[1].set_title('Age Dynamics (Diff from Previous Decade)')
    ax[1].set_ylabel('Difference (years)')
    ax[1].set_xlabel('Decade')
    ax[1].grid(linestyle='--', alpha=0.3)
    ax[1].legend()
    
    plt.tight_layout()
    path = os.path.join(output_dir, "age_statistics.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Age statistics report saved: {path}")

def plot_quality_heatmap(metrics, output_dir):
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Data Quality Metrics (Gini, Entropy, ENC)')
    
    path = os.path.join(output_dir, "heatmap.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Quality metrics heatmap saved: {path}")