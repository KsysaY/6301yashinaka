import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def file_reader(path, chunk_size=50000):
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
    full_df = pd.DataFrame()

    for chunk_df in data:
        full_df = pd.concat([full_df, chunk_df], ignore_index=True)

    grouped = full_df.groupby('Decade')['Age']

    stat = pd.DataFrame()
    stat['mean_age'] = grouped.mean()
    stat['std_age'] = grouped.std()
    stat['count'] = grouped.count()

    stat = stat.reset_index()
        
    stat['ci_95'] = 1.96 * stat['std_age'] / np.sqrt(stat['count'])
    stat['scatter_limit'] = 1.96 * stat['std_age']
        
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

    stat['age_diff'] = stat['mean_age'].diff()

    ax[1].plot(decades_str, stat['age_diff'], marker='*', color='orange', label='Изменение возраста')
    ax[1].axhline(0, color='black', linestyle='-')
    
    ax[1].set_title('Динамика')
    ax[1].set_ylabel('Разница')
    ax[1].set_xlabel('Десятилетие')
    
    ax[1].grid(axis='both', linestyle='--', alpha=0.5)
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = "MetObjects.csv"
    
    if os.path.exists(path):

        pipeline = data_aggregator(data_cleaner(file_reader(path)))

        result = next(pipeline)
        
        print("Результаты:")
        print(result.head(10))
        
        plot_results(result)
    else:
        print(f"Файл {path} не найден")