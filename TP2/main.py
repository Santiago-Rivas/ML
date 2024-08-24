import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mean(values):
    return sum(values) / len(values)


def linear_regression(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
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


# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('data/Advertising.csv')

# Shuffle the DataFrame rows
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Determine the split index
split_index = int(0.9 * len(shuffled_df))  # 80% for training

# Split the DataFrame into training and testing sets
train_df = shuffled_df[:split_index]
test_df = shuffled_df[split_index:]

TV_train = np.array(train_df['TV'])
Newspaper_train = np.array(train_df['Radio'])
Radio_train = np.array(train_df['Newspaper'])
Sales_train = np.array(train_df['Sales'])


TV_b_0, TV_b_1 = linear_regression(TV_train, Sales_train)
TV_y_pred = [x_test * TV_b_1 + TV_b_0 for x_test in np.array(test_df['TV'])]
TV_r_square = get_r_square(np.array(test_df['Sales']), TV_y_pred)
TV_MSE = mean_square_error(test_df['Sales'], TV_y_pred)
print("TV R^2\t\t", TV_r_square)
print("TV MES\t\t", TV_MSE)

Newspaper_b_0, Newspaper_b_1 = linear_regression(Newspaper_train, Sales_train)
Newspaper_y_pred = [x_test * Newspaper_b_1 + Newspaper_b_0 for x_test in np.array(test_df['Newspaper'])]
Newspaper_r_square = get_r_square(np.array(test_df['Sales']), Newspaper_y_pred)
Newspaper_MSE = mean_square_error(test_df['Sales'], Newspaper_y_pred)
print("Newspaper R^2\t", Newspaper_r_square)
print("Newspaper MES\t", Newspaper_MSE)

Radio_b_0, Radio_b_1 = linear_regression(Radio_train, Sales_train)
Radio_y_pred = [x_test * Radio_b_1 + Radio_b_0 for x_test in np.array(test_df['Radio'])]
Radio_r_square = get_r_square(np.array(test_df['Sales']), Radio_y_pred)
Radio_MSE = mean_square_error(test_df['Sales'], Radio_y_pred)
print("Radio R^2\t", Radio_r_square)
print("Radio MES\t", Radio_MSE)
exit()

# Calculo regresión múltiple
b = multiple_regression(np.column_stack((TV_train, Newspaper_train, Radio_train)), Sales_train)


# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

# Plot each variable against Sales
axes[0].scatter(TV, Sales, color='blue')
axes[0].set_title('Sales vs TV')
axes[0].set_xlabel('TV')
axes[0].set_ylabel('Sales')

axes[1].scatter(Radio, Sales, color='green')
axes[1].set_title('Sales vs Radio')
axes[1].set_xlabel('Radio')
axes[1].set_ylabel('Sales')

axes[2].scatter(Newspaper, Sales, color='red')
axes[2].set_title('Sales vs Newspaper')
axes[2].set_xlabel('Newspaper')
axes[2].set_ylabel('Sales')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig("./plt.png")
