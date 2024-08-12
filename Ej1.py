import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def calcular_regresion_lineal(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def plt_calorias_alcohol(df):
    # Define the bins and labels
    bins = [0, 1100, 1700, float('inf')]
    labels = ['CAT 1', 'CAT 2', 'CAT 3']

    # Create a new column in the DataFrame for the calorie category
    df['Calorie_Category'] = pd.cut(df['Calorías'], bins=bins, labels=labels, right=False)

    # Scatter plot
    plt.figure(figsize=(10, 6))
    categories = df['Calorie_Category'].unique()
    colors = ['blue', 'green', 'red']

    for category, color in zip(categories, colors):
        subset = df[df['Calorie_Category'] == category]
        plt.scatter(subset['Calorías'], subset['Alcohol'], label=category, color=color)

    plt.xlabel('Calorías')
    plt.ylabel('Alcohol')
    plt.title('Alcohol vs. Calorías')
    plt.legend(title='Categoría')
    plt.grid(True)
    plt.savefig("img/1_alcohol_calories_cat.png")


def plt_calorias_alcohol_sex(df):
    # Scatter plot
    plt.figure(figsize=(10, 6))
    colors = {'F': 'blue', 'M': 'red'}

    # Mapping Sexo values to full labels
    df['Sexo'] = df['Sexo'].map({'F': 'Female', 'M': 'Male'})

    # Scatter plot
    plt.figure(figsize=(10, 6))
    colors = {'Female': 'blue', 'Male': 'red'}

    for sexo in df['Sexo'].unique():
        subset = df[df['Sexo'] == sexo]
        plt.scatter(subset['Calorías'], subset['Alcohol'], label=sexo, color=colors[sexo])

    # plt.axvline(x=1100, color="black")
    plt.axvline(x=1400, color="black")
    plt.axvline(x=1700, color="black")
    plt.axvline(x=2100, color="black")

    plt.xlabel('Calorías')
    plt.ylabel('Alcohol')
    plt.title('Alcohol vs. Calorías')
    plt.legend(title='Sexo')
    plt.grid(True)
    plt.savefig("img/1_alcohol_calories_sex.png")


def plt_grasas_calorias(df):
    plt.figure(figsize=(10, 6))
    colors = {'F': 'blue', 'M': 'red'}
    df['Sexo'] = df['Sexo'].map({'F': 'Female', 'M': 'Male'})
    plt.figure(figsize=(10, 6))
    colors = {'Female': 'blue', 'Male': 'red'}

    for sexo in df['Sexo'].unique():
        subset = df[df['Sexo'] == sexo]
        plt.scatter(subset['Calorías'], subset['Grasas_sat'], label=sexo, color=colors[sexo])


    plt.xlabel('Calorías')
    plt.ylabel('Grasas_sat')
    plt.title('Grasas_sat vs. Calorías')
    plt.legend(title='Sex')
    plt.grid(True)
    plt.savefig("img/1_grasas_calories_sex.png")

def plt_grasas_calorias_reg(df):
    plt.figure(figsize=(10, 6))
    colors = {'F': 'blue', 'M': 'red'}
    df['Sexo'] = df['Sexo'].map({'F': 'Female', 'M': 'Male'})
    plt.figure(figsize=(10, 6))
    colors = {'Female': 'blue', 'Male': 'red'}

    regresion_params = {}

    for sexo in df['Sexo'].unique():
        subset = df[df['Sexo'] == sexo].copy()
        subset['Calorías'] = pd.to_numeric(subset['Calorías'], errors='coerce')
        subset['Grasas_sat'] = pd.to_numeric(subset['Grasas_sat'], errors='coerce')
        subset = subset.dropna(subset=['Calorías', 'Grasas_sat'])
        plt.scatter(subset['Calorías'], subset['Grasas_sat'], label=sexo, color=colors[sexo])

        # Calcular regresión lineal
        X = subset['Calorías'].values
        y = subset['Grasas_sat'].values
        X = X.astype(float)
        y = y.astype(float)
        m, c = calcular_regresion_lineal(X, y)
        y_pred = m * X + c
        sexo_name = sexo.replace(" ", "_")
        regresion_params[sexo_name] = (m, c)

        plt.plot(X, y_pred, color=colors[sexo], linestyle='--')

    plt.xlabel('Calorías')
    plt.ylabel('Grasas_sat')
    plt.title('Grasas_sat vs. Calorías')
    plt.legend(title='Sex')
    plt.grid(True)
    plt.savefig("img/1_grasas_calories_sex_reg.png")
    plt.close()

    # Convertir los parámetros a una matriz 2x2
    params_matrix = np.array([[regresion_params.get('Female', (np.nan, np.nan))[0],
                              regresion_params.get('Female', (np.nan, np.nan))[1]],
                             [regresion_params.get('Male', (np.nan, np.nan))[0],
                              regresion_params.get('Male', (np.nan, np.nan))[1]]])

    return params_matrix

def grasas_alcohol(df):
    df['Sexo'] = df['Sexo'].map({'F': 'Female', 'M': 'Male'})
    plt.figure(figsize=(10, 6))
    colors = {'Female': 'blue', 'Male': 'red'}

    for sexo in df['Sexo'].unique():
        subset = df[df['Sexo'] == sexo]
        plt.scatter(subset['Alcohol'], subset['Grasas_sat'], label=sexo, color=colors[sexo])

    plt.xlabel('Alcohol')
    plt.ylabel('Grasas Saturadas')
    plt.title('Grasas Saturadas vs. Alcohol')
    plt.legend(title='Sex')
    plt.grid(True)
    plt.savefig("img/1_alcohol_grasas.png")


def input_remove():
    file_path = "data/DatosAlimenticios.xls"

    df = pd.read_excel(file_path)
    #df = pd.read_csv("data/DatosAlimenticios.csv")

    df.replace(999.99, pd.NA, inplace=True)
    df = df.dropna()
    
    return df


def inferir_dato(calorias, sexo, params_matrix):
    if sexo == 'F':
        row = 0
    elif sexo == 'M':
        row = 1
    else:
        print("Sexo no válido")
    
    m = params_matrix[row, 0]
    c = params_matrix[row, 1]
    return m * calorias + c


def plt_calorias_alcohol_reg(df):
    df['Sexo'] = df['Sexo'].map({'F': 'Female', 'M': 'Male'})

    # Definir las categorías de calorías
    def categorize_calories(row):
        if row['Calorías'] < 1400:
            return 'CAT 1'
        elif 1400 <= row['Calorías'] <= 1700:
            return 'CAT 2'
        else:
            return 'CAT 3'

    # Crear una nueva columna 'Calorías_cat' para las categorías
    df['Calorías_cat'] = df.apply(categorize_calories, axis=1)

    # Colores para las combinaciones de sexo y categorías de calorías
    colors = {
        ('Female', 'CAT 1'): 'blue',
        ('Female', 'CAT 2'): 'cyan',
        ('Female', 'CAT 3'): 'lightblue',
        ('Male', 'CAT 1'): 'red',
        ('Male', 'CAT 2'): 'orange',
        ('Male', 'CAT 3'): 'salmon'
    }

    plt.figure(figsize=(10, 6))

    # Graficar cada conjunto con un color diferente y calcular la regresión lineal
    for (sexo, calorías_cat), subset in df.groupby(['Sexo', 'Calorías_cat']):
        # Asegurarse de que los datos sean numéricos y eliminar NaNs
        X = pd.to_numeric(subset['Calorías'], errors='coerce').dropna().values
        Y = pd.to_numeric(subset['Alcohol'], errors='coerce').dropna().values

        if len(X) > 1 and len(Y) > 1:  # Asegurarse de que haya suficientes datos para la regresión
            plt.scatter(X, Y, 
                        label=f'{sexo} ({calorías_cat})', 
                        color=colors[(sexo, calorías_cat)])

            # Calcular la regresión lineal
            coeffs = np.polyfit(X, Y, 1)  # coeficientes de la recta
            slope, intercept = coeffs

            # Imprimir los valores de la regresión
            print(f"Regresión lineal para {sexo} ({calorías_cat}):")
            print(f"  Pendiente: {slope}")
            print(f"  Ordenada: {intercept}\n")

            # Graficar la línea de regresión
            plt.plot(X, slope * X + intercept, color=colors[(sexo, calorías_cat)], linestyle='--')

    # Líneas verticales
    plt.axvline(x=1400, color="black")
    plt.axvline(x=1700, color="black")

    # Configuración de la gráfica
    plt.xlabel('Calorías')
    plt.ylabel('Alcohol')
    plt.title('Alcohol vs. Calorías')
    plt.legend(title='Category')
    plt.grid(True)

    # Guardar la imagen
    plt.savefig("img/1_alcohol_calories_reg.png")


def main():
    df = input_remove()
    plt_calorias_alcohol(df.copy())
    plt_calorias_alcohol_sex(df.copy())
    plt_grasas_calorias(df.copy())
    grasas_alcohol(df.copy())
    params_matrix = plt_grasas_calorias_reg(df.copy())
    calorias = 1000
    sexo = 'F'
    salida = inferir_dato(calorias, sexo, params_matrix)
    print("Con " + str(calorias) + " calorias y " + sexo + ": " + str(salida) + " grasas") 

    plt_calorias_alcohol_reg(df.copy())



if __name__ == "__main__":
    main()
