import numpy as np
import pandas as pd

INPUT_LENGTH = 20 

def read_input():
    df = pd.read_csv('data/german_credit.csv')

    columna_objetivo = df['Creditability']
    otras_columnas = df.drop(columns=['Creditability'])

    return otras_columnas, columna_objetivo

def p_f(x, beta):
    x = np.insert(x, 0, 1) # en la posicion 0 poner 1 (por el B0)
    print(len(beta))
    print(len(x))
    print(x)
    return np.exp(-np.dot(x, beta)) / (1 + np.exp(-np.dot(x, beta)))

def L(beta):
    x, y = read_input()
    salida = 1.0
    for i in range(len(y)):
        print(x.iloc[i])
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

def maximize_beta():
    delta = 0.01
    iter = 1000
    beta = np.random.rand(INPUT_LENGTH + 1)
    x, y = read_input()
    x_mat = np.array(x)
    ones_column = np.ones((x_mat.shape[0], 1))
    x_mat = np.hstack((ones_column, x_mat))
    for _ in range(iter):
        beta += log_likelihood_gradient(x_mat, y, beta)
    L(beta)
    return beta


def main():
    print(maximize_beta())

if __name__ == '__main__':
    main()

