import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def feature_scaling(x):
    x_min = min(x)
    x_max = max(x)
    return [(x_val - x_min) / (x_max - x_min) for x_val in x]


def linear_regression(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    b_1 = sum((xi - mean_x) * (yi - mean_y)
              for xi, yi in zip(x, y)) / sum((xi - mean_x) ** 2 for xi in x)
    b_0 = mean_y - b_1 * mean_x
    return b_0, b_1


def multiple_regression(X, y):
    columna_uno = np.ones((X.shape[0], 1))
    X_matrix = np.column_stack((columna_uno, X))
    return np.dot(np.dot(np.linalg.inv(np.dot(X_matrix.T, X_matrix)), X_matrix.T), y)


def get_r_square(y_test, y_pred):
    y_mean = np.mean(y_test)
    a = [(y_pre - y_val) ** 2 for y_pre, y_val in zip(y_pred, y_test)]
    b = [(y_val - y_mean) ** 2 for y_val in y_test]
    return 1 - (np.sum(a)/np.sum(b))


def mean_square_error(y_test, y_predicted):
    n = len(y_test)
    return np.sum([(y_t - y_p) ** 2 for y_t, y_p in zip(y_test, y_predicted)])/n

def mean_absolute_error(y_test, y_predicted):
    n = len(y_test)
    return np.sum([abs(y_t - y_p) for y_t, y_p in zip(y_test, y_predicted)])/n


def get_adjusted_r_square(r_square, n, q):
    return 1 - ((1 - r_square) * (n - 1) / (n - q))


def fisher_test(y_test, y_pred, X):

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    n = len(y_test)  # Número de observaciones
    p = X.shape[1]   # Número de variables independientes (incluyendo la constante)
    
    y_mean = np.mean(y_test)
    
    TSS = np.sum((y_test - y_mean) ** 2)
    
    # Suma de cuadrados del error (SSE)
    RSS = np.sum((y_test - y_pred) ** 2)
    
    df_res = n-p-1
    
    F_statistic = ((TSS - RSS)/ p) / (RSS / df_res)
    
    p_value = 1 - stats.f.cdf(F_statistic, p, df_res)

    
    return F_statistic, p_value

def plot_comparison(array_1, array_2):
    x = list(range(1, 21))

    plt.figure(figsize=(10, 6))
    plt.scatter(x, array_1, color='blue', label='Sales test')
    plt.scatter(x, array_2, color='red', label='Predicted')
   
    plt.legend()
    plt.xticks([])

    plt.savefig(f"output/plot_multiple.png")

def plot_real_vs_reg(x_test, y_test, b_0, b_1, variable_name):
    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Scatter plot of the data
    plt.scatter(x_test, y_test, color='blue', label='Data')

    # Create x values for the line plot
    x_line = np.linspace(min(x_test), max(x_test), 100)

    # Calculate the y values using the linear equation y = b_1 * x + b_0
    y_line = b_1 * x_line + b_0

    # Plot the linear line
    plt.plot(x_line, y_line, color='red', linestyle='--', label='Linear Fit')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Scatter Plot with Linear Fit {variable_name}')
    plt.legend()
    plt.savefig(f"output/plot_{variable_name}.png")


# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('data/Advertising.csv')

# Shuffle the DataFrame rows
shuffled_df = df.sample(frac=1, random_state=1).reset_index(drop=True)

# Determine the split index
split_index = int(0.9 * len(shuffled_df))  # 80% for training

# Split the DataFrame into training and testing sets
train_df = shuffled_df[:split_index]
test_df = shuffled_df[split_index:]

TV_train = np.array(train_df['TV'])
Newspaper_train = np.array(train_df['Radio'])
Radio_train = np.array(train_df['Newspaper'])
Sales_train = np.array(train_df['Sales'])

TV_test = np.array(test_df['TV'])
Newspaper_test = np.array(test_df['Radio'])
Radio_test = np.array(test_df['Newspaper'])
Sales_test = np.array(test_df['Sales'])


TV_b_0, TV_b_1 = linear_regression(TV_train, Sales_train)
TV_y_pred = [x_test * TV_b_1 + TV_b_0 for x_test in np.array(test_df['TV'])]
TV_r_square = get_r_square(np.array(Sales_test), TV_y_pred)
TV_MSE = mean_square_error(test_df['Sales'], TV_y_pred)
TV_MAE = mean_absolute_error(test_df['Sales'], TV_y_pred)

plot_real_vs_reg(TV_test, Sales_test, TV_b_0, TV_b_1, "TV")

print("TV R^2\t\t", TV_r_square)
print("TV MSE\t\t", TV_MSE)
print("TV MAE\t\t", TV_MAE)

Newspaper_b_0, Newspaper_b_1 = linear_regression(Newspaper_train, Sales_train)
Newspaper_y_pred = [x_test * Newspaper_b_1 +
                    Newspaper_b_0 for x_test in np.array(test_df['Newspaper'])]

plot_real_vs_reg(Newspaper_test, Sales_test,
                 Newspaper_b_0, Newspaper_b_1, "Newspaper")

Newspaper_r_square = get_r_square(np.array(test_df['Sales']), Newspaper_y_pred)
Newspaper_MSE = mean_square_error(test_df['Sales'], Newspaper_y_pred)
Newspaper_MAE = mean_absolute_error(test_df['Sales'], Newspaper_y_pred)
print("Newspaper R^2\t", Newspaper_r_square)
print("Newspaper MSE\t", Newspaper_MSE)
print("Newspaper MAE\t", Newspaper_MAE)

Radio_b_0, Radio_b_1 = linear_regression(Radio_train, Sales_train)
Radio_y_pred = [x_test * Radio_b_1 +
                Radio_b_0 for x_test in np.array(test_df['Radio'])]

plot_real_vs_reg(Radio_test, Sales_test, Radio_b_0, Radio_b_1, "Radio")

Radio_r_square = get_r_square(np.array(test_df['Sales']), Radio_y_pred)
Radio_MSE = mean_square_error(test_df['Sales'], Radio_y_pred)
Radio_MAE = mean_absolute_error(test_df['Sales'], Radio_y_pred)
print("Radio R^2\t", Radio_r_square)
print("Radio MSE\t", Radio_MSE)
print("Radio MAE\t", Radio_MAE)

print()
TV_train = feature_scaling(np.array(train_df['TV']))
Newspaper_train = feature_scaling(np.array(train_df['Radio']))
Radio_train = feature_scaling(np.array(train_df['Newspaper']))
Sales_train = feature_scaling(np.array(train_df['Sales']))

TV_test = feature_scaling(np.array(test_df['TV']))
Newspaper_test = feature_scaling(np.array(test_df['Radio']))
Radio_test = feature_scaling(np.array(test_df['Newspaper']))
Sales_test = feature_scaling(np.array(test_df['Sales']))

# Calculo regresión múltiple
X = np.column_stack(
    (TV_train, Newspaper_train, Radio_train))
b = multiple_regression(X, Sales_train)
y_pred = [b[0] + b[1] * tv + b[2] * newspaper + b[3] * radio for tv,
          newspaper, radio in zip(TV_test, Newspaper_test, Radio_test)]
multi_reg_r_square = get_r_square(Sales_test, y_pred)
multi_reg_adjusted_r_square = get_adjusted_r_square(
    multi_reg_r_square, len(y_pred), len(b))
multi_reg_MSE = mean_square_error(Sales_test, y_pred)
print("Multi Reg R^2\t\t", multi_reg_r_square)
print("Multi Reg Adj R^2\t", multi_reg_adjusted_r_square)
print("Multi Reg MSE\t\t", multi_reg_MSE)
print("β Coefs\t\t\t", b)

plot_comparison(Sales_test, y_pred)

F_statistic, p_value = fisher_test(Sales_test, y_pred, X)

print("F: " + str(F_statistic))
print("p_value: " + str(p_value))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Graficar cada variable contra Sales
axes[0, 0].scatter(shuffled_df["TV"], shuffled_df["Sales"], color='blue')
axes[0, 0].set_title('Sales vs TV')
axes[0, 0].set_xlabel('TV')
axes[0, 0].set_ylabel('Sales')

axes[0, 1].scatter(shuffled_df["Radio"], shuffled_df["Sales"], color='green')
axes[0, 1].set_title('Sales vs Radio')
axes[0, 1].set_xlabel('Radio')
axes[0, 1].set_ylabel('Sales')

axes[0, 2].scatter(shuffled_df["Newspaper"], shuffled_df["Sales"], color='red')
axes[0, 2].set_title('Sales vs Newspaper')
axes[0, 2].set_xlabel('Newspaper')
axes[0, 2].set_ylabel('Sales')

# Graficar las relaciones entre las otras variables
axes[1, 0].scatter(shuffled_df["TV"], shuffled_df["Radio"], color='purple')
axes[1, 0].set_title('Radio vs TV')
axes[1, 0].set_xlabel('TV')
axes[1, 0].set_ylabel('Radio')

axes[1, 1].scatter(shuffled_df["TV"], shuffled_df["Newspaper"], color='orange')
axes[1, 1].set_title('Newspaper vs TV')
axes[1, 1].set_xlabel('TV')
axes[1, 1].set_ylabel('Newspaper')

axes[1, 2].scatter(shuffled_df["Radio"], shuffled_df["Newspaper"], color='brown')
axes[1, 2].set_title('Newspaper vs Radio')
axes[1, 2].set_xlabel('Radio')
axes[1, 2].set_ylabel('Newspaper')

# Ajustar el layout para evitar solapamientos
plt.tight_layout()

# Guardar la gráfica
plt.savefig("./output/plt.png")