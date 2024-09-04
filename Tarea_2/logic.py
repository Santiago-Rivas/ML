import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize

INPUT_LENGTH = 20 

def read_input_random():
    df = pd.read_csv('data/german_credit.csv')

    shuffled_df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    split_index = int(0.8 * len(shuffled_df))  # 80% for training

    # Split the DataFrame into training and testing sets
    train_df = shuffled_df[:split_index]
    test_df = shuffled_df[split_index:]

    return train_df, test_df


def split_df(df):
    columna_objetivo = df['Creditability']
    otras_columnas = df.drop(columns=['Creditability'])

    return otras_columnas, columna_objetivo


def p_f(x, beta):
    x = np.insert(x, 0, 1) # en la posicion 0 poner 1 (por el B0)
    return np.exp(-np.dot(x, beta)) / (1 + np.exp(-np.dot(x, beta)))

def L(beta, x, y):
    salida = 1.0
    for i in range(len(y)):
        #print(x.iloc[i])
        p = p_f(x.iloc[i], beta)
        if y.iloc[i] == 1:
            salida *= p
        else:
            salida *= (1 - p)
    return salida

def log_likelihood_gradient(X, y, beta):
    aux = np.exp(-np.dot(X, beta))
    p = aux / (1 + aux)
    
    gradient = np.dot(X.T, y - p)
    return gradient

def maximize_beta(x, y):
    delta = 0.0001
    iter =  10000
    beta = np.zeros(INPUT_LENGTH + 1)
    x_mat = np.array(x)
    ones_column = np.ones((x_mat.shape[0], 1))
    x_mat = np.hstack((ones_column, x_mat))
    for index in range(iter):
        gradient = log_likelihood_gradient(x_mat, y, beta)
        beta -= delta * gradient
        if index % 1000 == 0:
            print(index)
            print(gradient.dot(gradient))
        if gradient.dot(gradient) < 0.0001:
            break
    L(beta, x, y)
    return beta

def feature_scaling(x):
    x_min = min(x)
    x_max = max(x)
    return [(x_val - x_min) / (x_max - x_min) for x_val in x]

def input_scaling(df):
    Account = feature_scaling(np.array(df['Account Balance']))
    Duration = feature_scaling(np.array(df['Duration of Credit (month)']))
    Payment = feature_scaling(np.array(df['Payment Status of Previous Credit']))
    Purpose = feature_scaling(np.array(df['Purpose']))
    Value = feature_scaling(np.array(df['Value Savings/Stocks']))
    Amount = feature_scaling(np.array(df['Credit Amount']))
    Length = feature_scaling(np.array(df['Length of current employment']))
    Instalment = feature_scaling(np.array(df['Instalment per cent']))
    Marital = feature_scaling(np.array(df['Sex & Marital Status']))
    Guarantors = feature_scaling(np.array(df['Guarantors']))
    Adress = feature_scaling(np.array(df['Duration in Current address']))
    Asset = feature_scaling(np.array(df['Most valuable available asset']))
    Age = feature_scaling(np.array(df['Age (years)']))
    Concurrent = feature_scaling(np.array(df['Concurrent Credits']))
    Apartment = feature_scaling(np.array(df['Type of apartment']))
    Bank = feature_scaling(np.array(df['No of Credits at this Bank']))
    Occupation = feature_scaling(np.array(df['Occupation']))
    Departament = feature_scaling(np.array(df['No of dependents']))
    Telephone = feature_scaling(np.array(df['Telephone']))
    Worker = feature_scaling(np.array(df['Foreign Worker']))

    combined = np.column_stack((Account, Duration, Payment, Purpose, Value, Amount, Length, Instalment, Marital, Guarantors, Adress, Asset, Age, Concurrent, Apartment, Bank, Occupation, Departament, Telephone, Worker))
    column_names = ["Account Balance", "Duration of Credit (month)", "Payment Status of Previous Credit", "Purpose",
                    "Value Savings/Stocks", "Credit Amount", "Length of current employment", "Instalment per cent", 
                    "Sex & Marital Status", "Guarantors", "Duration in Current address", "Most valuable available asset",
                    "Age (years)", "Concurrent Credits", "Type of apartment", 
                    "No of Credits at this Bank", "Occupation", "No of dependents", "Telephone", "Foreign Worker"]

    df_scaling = pd.DataFrame(combined, columns=column_names)
    return df_scaling


def plot_matriz(TP, FP, FN, TN):

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    conf_matrix = np.array([[TP, FP],
                        [FN, TN]])

    # Etiquetas para el gráfico
    labels = np.array([[TP, FP], [FN, TN]])

    # Graficar la matriz de confusión
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=labels, fmt="", cmap="Blues", cbar=False, square=True, linewidths=0.5, linecolor='black')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



def matriz_confusion(x, y, beta):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        p = p_f(x.iloc[i], beta)
        if p >= 0.5:
            if y.iloc[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y.iloc[i] == 0:
                TN += 1
            else:
                FN += 1
    print("TP: ", TP)
    print("TN: ", TN)
    print("FP: ", FP)
    print("FN: ", FN)

    plot_matriz(TP, FP, FN, TN)


def read_input_balance():
    df = pd.read_csv('data/german_credit.csv')
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # Separar los valores con Creditability = 1 y Creditability = 0
    df_1 = shuffled_df[shuffled_df['Creditability'] == 1]
    df_0 = shuffled_df[shuffled_df['Creditability'] == 0]
    #print(len(df_1), len(df_0))
    if(len(df_1) > len(df_0)):
        max_sample = int(len(df_0) * 0.2)
    else:
        max_sample = int(len(df_1) * 0.2)
    # Seleccionar 40 ejemplos de cada grupo para el conjunto de prueba
    test_df_1 = df_1.sample(n=max_sample, random_state=1)
    test_df_0 = df_0.sample(n=max_sample, random_state=1)

    # Combinar ambos conjuntos para formar el conjunto de prueba
    test_df = pd.concat([test_df_1, test_df_0]).reset_index(drop=True)

    # Eliminar las filas seleccionadas para el conjunto de prueba del conjunto de entrenamiento
    train_df = pd.concat([df_1.drop(test_df_1.index), df_0.drop(test_df_0.index)]).reset_index(drop=True)

    return train_df, test_df


def main():
    #train_df, test_df = read_input_random()
    train_df, test_df = read_input_balance()
    x, y = split_df(train_df)
    x_scaling = input_scaling(x)

    beta = maximize_beta(x_scaling, y)
    print(beta)
    
    x_test, y_test = split_df(test_df)
    x_test_scaling = input_scaling(x_test)
    matriz_confusion(x_test_scaling, y_test, beta)

if __name__ == '__main__':
    main()

