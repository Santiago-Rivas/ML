import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data/DatosAlimenticios.xls"
df = pd.read_excel(file_path)
original_df = df
# TODO: CAMBIAR Y USAR REGRESIONES
df.replace(999.99, pd.NA, inplace=True)
df = df.dropna()


# Boxplot para 'Grasas' por sexo
plt.figure(figsize=(10, 6))
fig, ax = plt.subplots()
box = df.boxplot(column='Grasas_sat', by='Sexo', ax=ax, grid=False, patch_artist=True,
                 boxprops=dict(color='black', facecolor='white'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 flierprops=dict(color='black', markeredgecolor='black'),
                 medianprops=dict(color='red'),
                 meanline=False)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xticklabels(['Femenino', 'Masculino'])
plt.title('Distribución de Grasas Saturadas por Sexo')
plt.suptitle('')
plt.xlabel('Sexo')
plt.ylabel('Grasas Saturadas')
plt.savefig("img/3_GrasasSaturadas_Sexo.png")

# Boxplot para 'Calorías' por sexo
plt.figure(figsize=(10, 6))
fig, ax = plt.subplots()
box = df.boxplot(column='Calorías', by='Sexo', ax=ax, grid=False, patch_artist=True,
                 boxprops=dict(color='black', facecolor='white'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 flierprops=dict(color='black', markeredgecolor='black'),
                 medianprops=dict(color='red'),
                 meanline=False)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xticklabels(['Femenino', 'Masculino'])
plt.title('Distribución de Calorias por Sexo')
plt.suptitle('')
plt.xlabel('Sexo')
plt.ylabel('Calorías')
plt.savefig("img/3_Calorías_Sexo.png")

# Boxplot para 'Alcohol' por sexo
plt.figure(figsize=(10, 6))
fig, ax = plt.subplots()
box = df.boxplot(column='Alcohol', by='Sexo', ax=ax, grid=False, patch_artist=True,
                 boxprops=dict(color='black', facecolor='white'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 flierprops=dict(color='black', markeredgecolor='black'),
                 medianprops=dict(color='red'),
                 meanline=False)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xticklabels(['Femenino', 'Masculino'])
plt.title('Distribución de Alcohol por Sexo')
plt.suptitle('')
plt.xlabel('Sexo')
plt.ylabel('Alcohol')
plt.savefig("img/3_Alcohol_Sexo.png")