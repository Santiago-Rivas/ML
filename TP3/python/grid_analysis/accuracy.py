import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo CSV
df = pd.read_csv('./output/new2/metrics_final.csv', sep=';')

# Agrupar por 'N' y calcular la media y el error estándar (std) de 'accuracy'
accuracy_mean = df.groupby('N')['accuracy'].mean()
accuracy_std = df.groupby('N')['accuracy'].std()

# Graficar con barras de error
plt.errorbar(accuracy_mean.index, accuracy_mean, yerr=accuracy_std, fmt='-o', capsize=5, label='Accuracy')

# Etiquetas y título
plt.xlabel('Tamño grilla pixeles')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Tamño grilla pixeles')
plt.grid(False)
plt.show()