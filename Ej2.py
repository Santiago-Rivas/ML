import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data/DatosAlimenticios_cambiados.csv"
df = pd.read_csv(file_path)
original_df = df

# Boxplot for 'Grasas_sat'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Grasas_sat'])
plt.title('Boxplot for Grasas Sat')
plt.savefig("img/2_grasas.png")
plt.close()

# Boxplot for 'Alcohol'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Alcohol'])
plt.title('Boxplot for Alcohol')
plt.savefig("img/2_alcohol.png")
plt.close()

# Boxplot for 'Calorías'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Calorías'])
plt.title('Boxplot for Calorías')
plt.savefig("img/2_calorias.png")
plt.close()
