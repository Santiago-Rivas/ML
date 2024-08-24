import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mean(values):
    return sum(values) / len(values)

def linear_regression(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    b_1 = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / sum((xi - mean_x) ** 2 for xi in x)
    b_0 = mean_y - b_1 * mean_x
    return b_0, b_1

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('Advertising.csv')

# Shuffle the DataFrame rows
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Determine the split index
split_index = int(0.8 * len(shuffled_df))  # 80% for training

# Split the DataFrame into training and testing sets
train_df = shuffled_df[:split_index]
test_df = shuffled_df[split_index:]

TV = np.array(df['TV'])
Newspaper = np.array(df['Radio'])
Radio = np.array(df['Newspaper'])
Sales = np.array(df['Sales'])

print(linear_regression(TV, Sales))
print(linear_regression(Newspaper, Sales))
print(linear_regression(Radio, Sales))


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