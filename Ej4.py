import pandas as pd
import matplotlib.pyplot as plt

def boxplot_cat(df):
    bins = [0, 1100, 1700, float('inf')]
    labels = ['CATE 1', 'CATE 2', 'CATE 3']
    df['Calorías_Categoría'] = pd.cut(df['Calorías'], bins=bins, labels=labels)

    plt.figure(figsize=(10, 6))
    # Configuración del gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear el boxplot con el formato especificado
    box = df.boxplot(column='Alcohol', by='Calorías_Categoría', ax=ax, grid=False, patch_artist=True,
                    boxprops=dict(color='black', facecolor='white'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(color='black', markeredgecolor='black'),
                    medianprops=dict(color='red'),
                    meanline=False)

    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title('Consumo de Alcohol según Categorías de Calorías')
    plt.suptitle('')
    plt.xlabel('Categorías de Calorías')
    plt.ylabel('Consumo de Alcohol')
    plt.savefig("img/4_alcohol_calories_box.png")


def scatter_cat(df):
    bins = [0, 1100, 1700, float('inf')]
    labels = ['CAT 1', 'CAT 2', 'CAT 3']
    df['Calorie_Category'] = pd.cut(df['Calorías'], bins=bins, labels=labels, right=False)

    # Scatter plot
    plt.figure(figsize=(10, 6))
    categories = df['Calorie_Category'].unique()
    colors = ['blue', 'green', 'red']

    for category, color in zip(categories, colors):
        subset = df[df['Calorie_Category'] == category]
        plt.scatter(subset['Calorías'], subset['Alcohol'], label=category, color=color)

    plt.xlabel('Calorías')
    plt.ylabel('Consumo de Alcohol')
    plt.title('Consumo de Alcohol según Categorías de Calorías')
    plt.legend(title='Categoría')
    plt.grid(True)
    plt.savefig("img/4_alcohol_calories_cat.png")


def input_remove():
    file_path = "data/DatosAlimenticios.xls"
    df = pd.read_excel(file_path)
    df.replace(999.99, pd.NA, inplace=True)
    df = df.dropna()
    
    return df

def main():
    df = input_remove()
    boxplot_cat(df.copy())
    scatter_cat(df.copy())

if __name__ == "__main__":
    main()
