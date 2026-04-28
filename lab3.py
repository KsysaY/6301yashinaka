import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def file_reader(path, chunk_size=1000):
    """Чтение файла"""
    cols = ['AccessionYear', 'Object Begin Date']
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False, usecols=cols):
        yield chunk

def data_cleaner(data):
    """Фильтрация"""
    for df_with_nan in data:
        df_with_nan['AccessionYear'] = pd.to_numeric(df_with_nan['AccessionYear'], errors='coerce')
        
        df = df_with_nan.dropna().copy()
        
        df['Age'] = df['AccessionYear'] - df['Object Begin Date']
        df['Decade'] = (df['AccessionYear'] // 10) * 10
        
        yield df[['Decade', 'Age']]

def data_aggregator(data):
    """Агрегация"""
    stat = pd.DataFrame()

    for chunk_df in data:
        chunk_info = pd.DataFrame({
            'size': chunk_df.groupby('Decade')['Age'].size(),
            'sum': chunk_df.groupby('Decade')['Age'].sum(),
            'sum_sq': (chunk_df['Age']**2).groupby(chunk_df['Decade']).sum()
        })
        stat = pd.concat([stat, chunk_info])

    stat_summary = stat.groupby('Decade').sum()

    size = stat_summary['size']
    sum_val = stat_summary['sum']
    
    stat_summary['mean_age'] = sum_val / size
    
    var = (stat_summary['sum_sq'] - (sum_val**2 / size)) / (size - 1)
    stat_summary['std_age'] = np.sqrt(var.clip(lower=0))
    
    stat = stat_summary.reset_index().sort_values('Decade')
    
    stat['ci_95'] = 1.96 * stat['std_age'] / np.sqrt(stat['size']) 
    stat['scatter_limit'] = 1.96 * stat['std_age']             
    stat['age_diff'] = stat['mean_age'].diff()
    
    yield stat
  

def plot_results(stat):
    
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    
    decades_str = stat['Decade'].astype(int).astype(str)

    low_scatter = np.minimum(stat['mean_age'], stat['scatter_limit'])
    
    ax[0].bar(decades_str, stat['mean_age'], color='blue', label='Средний возраст')
    
    ax[0].errorbar(decades_str, stat['mean_age'], yerr=[low_scatter, stat['scatter_limit']], color='gray', fmt='none', label='Рассеяние')
    ax[0].errorbar(decades_str, stat['mean_age'], yerr=stat['ci_95'], color='black', capsize=5, fmt='none', label='95% ДИ')
    
    ax[0].set_title('Средний возраст объектов')
    ax[0].set_ylabel('Возраст')
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)

    ax[1].plot(decades_str, stat['age_diff'], marker='*', color='orange', label='Изменение возраста')
    ax[1].axhline(0, color='black', linestyle='-')
    
    ax[1].set_title('Динамика')
    ax[1].set_ylabel('Разница')
    ax[1].set_xlabel('Десятилетие')
    
    ax[1].grid(axis='both', linestyle='--', alpha=0.5)
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

def frequency(path, chunk_size=1000):
    col = ['Department', 'Culture', 'Medium', 'Classification', 'Country']
    total_counts = pd.DataFrame(columns=['field', 'value', 'count'])
    
    for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=col, low_memory=False):
        for field in col:
            counts = chunk[field].dropna().astype(str).value_counts().reset_index()
            counts.columns = ['value', 'count']
            counts['field'] = field
            total_counts = pd.concat([total_counts, counts], ignore_index=True)
    
    result = total_counts.groupby(['field', 'value'])['count'].sum().unstack(level=0)
    result = result.fillna(0).astype(int)
    
    return result

def calculate_metrics(df):
    results = pd.DataFrame()
    
    for field in df.columns:
        counts = df[field].values
        counts = counts[counts > 0]
        
        if len(counts) == 0:
            continue
        
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

def plot_quality_heatmap(metrics):
    """Построение тепловой карты"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics, annot=True, cmap='viridis', vmin=0, vmax=1)
    plt.title('Метрики')
    plt.show()

if __name__ == "__main__":
    path = "MetObjects.csv"
    
    if os.path.exists(path):

        pipeline = data_aggregator(data_cleaner(file_reader(path)))

        result = next(pipeline)
        
        print("Результаты:")
        print(result.head(10))
        
        plot_results(result)

        #Допы
        counts = frequency(path)
        metrics = calculate_metrics(counts)
        plot_quality_heatmap(metrics)
    else:
        print(f"Файл {path} не найден")