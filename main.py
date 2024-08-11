import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data/DatosAlimenticios.xls"

df = pd.read_excel(file_path)
#df = pd.read_csv("data/DatosAlimenticios.csv")

original_df = df

df.replace(999.99, pd.NA, inplace=True)
df = df.dropna()


# Boxplot for 'Grasas_sat'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Grasas_sat'])
plt.title('Boxplot for Grasas Sat')
plt.savefig("img/boxplot_grasas_sat.png")
plt.close()

# Boxplot for 'Alcohol'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Alcohol'])
plt.title('Boxplot for Alcohol')
plt.savefig("img/boxplot_alcohol.png")
plt.close()

# Boxplot for 'Calorías'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Calorías'])
plt.title('Boxplot for Calorías')
plt.savefig("img/boxplot_calorias.png")
plt.close()

# Punto 2

# Boxplot para 'Grasas_sat' por sexo
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
plt.savefig("img/2_GrasasSaturadas_Sexo.png")

# Boxplot para 'Calorías' por sexo
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
plt.savefig("img/2_Calorías_Sexo.png")

# Boxplot para 'Alcohol' por sexo
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
plt.savefig("img/2_Alcohol_Sexo.png")

df_males = df[df['Sexo'] == 'M']
df_females = df[df['Sexo'] == 'F']

# Boxplot for 'Grasas_sat'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df_males['Grasas_sat'])
plt.title('Boxplot for Grasas Sat (Males)')
plt.savefig("img/boxplot_grasas_sat_males.png")
plt.close()

# Boxplot for 'Alcohol'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df_males['Alcohol'])
plt.title('Boxplot for Alcohol (Males)')
plt.savefig("img/boxplot_alcohol_males.png")
plt.close()

# Boxplot for 'Calorías'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df_males['Calorías'])
plt.title('Boxplot for Calorías (Males)')
plt.savefig("img/boxplot_calorias_males.png")
plt.close()

# Boxplot for 'Grasas_sat'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df_females['Grasas_sat'])
plt.title('Boxplot for Grasas Sat (Females)')
plt.savefig("img/boxplot_grasas_sat_females.png")
plt.close()

# Boxplot for 'Alcohol'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df_females['Alcohol'])
plt.title('Boxplot for Alcohol (Females)')
plt.savefig("img/boxplot_alcohol_females.png")
plt.close()

# Boxplot for 'Calorías'
plt.figure(figsize=(6, 4))
sns.boxplot(y=df_females['Calorías'])
plt.title('Boxplot for Calorías (Females)')
plt.savefig("img/boxplot_calorias_females.png")
plt.close()

bins = [0, 1100, 1700, float('inf')]
labels = ['Cate 1', 'Cate 2', 'Cate 3']

df_cate = df.copy()
df_cate.loc[:, 'Calorie_Category'] = pd.cut(df_cate['Calorías'],
                                            bins=bins,
                                            labels=labels,
                                            right=False)

hist_data = df_cate.groupby('Calorie_Category')['Alcohol'].sum()

plt.figure(figsize=(12, 10))
hist_data.plot(kind='bar', color='skyblue')
plt.xlabel('Categoria de Calorias')
plt.ylabel('Alcohol Consumido')
plt.title('Alcohol Consumido por Categoria de Calorias')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("img/Alcohol_per_Cate.png")
plt.close()


# Define the bins and labels
bins = [0, 1100, 1700, float('inf')]
labels = ['1100 or fewer', '1101-1700', '1701 or more']

# Create a new column in the DataFrame for the calorie category
df_cop = df.copy()  # Ensure we're working with a copy of the DataFrame
df = df.copy()  # Ensure we're working with a copy of the DataFrame
df['Calorie_Category'] = pd.cut(df['Calorías'], bins=bins, labels=labels, right=False)

# Scatter plot
plt.figure(figsize=(10, 6))
categories = df['Calorie_Category'].unique()
colors = ['blue', 'green', 'red']

for category, color in zip(categories, colors):
    subset = df[df['Calorie_Category'] == category]
    plt.scatter(subset['Calorías'], subset['Alcohol'], label=category, color=color)

plt.xlabel('Calorías')
plt.ylabel('Alcohol Consumption')
plt.title('Scatter Plot of Alcohol Consumption vs. Calorías')
plt.legend(title='Calorie Category')
plt.grid(True)
plt.savefig("img/scatter_alcohol_vs_calories_cate.png")

df = df_cop.copy()

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
plt.ylabel('Alcohol Consumption')
plt.title('Scatter Plot of Alcohol Consumption vs. Calorías Colored by Sex')
plt.legend(title='Sex')
plt.grid(True)
plt.savefig("img/scatter_alcohol_vs_calories_sex.png")

df = df_cop.copy()

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
    plt.scatter(subset['Calorías'], subset['Grasas_sat'], label=sexo, color=colors[sexo])

plt.axvline(x=1700, color="black")
plt.axvline(x=1100, color="black")
plt.xlabel('Calorías')
plt.ylabel('Grasas_sat Consumption')
plt.title('Scatter Plot of Grasas_sat Consumption vs. Calorías Colored by Sex')
plt.legend(title='Sex')
plt.grid(True)
plt.savefig("img/scatter_grasas_sat_vs_calories_sex.png")

df = df_cop.copy()

# Mapping Sexo values to full labels
df['Sexo'] = df['Sexo'].map({'F': 'Female', 'M': 'Male'})

# Scatter plot
plt.figure(figsize=(10, 6))
colors = {'Female': 'blue', 'Male': 'red'}

for sexo in df['Sexo'].unique():
    subset = df[df['Sexo'] == sexo]
    plt.scatter(subset['Alcohol'], subset['Grasas_sat'], label=sexo, color=colors[sexo])

plt.xlabel('Alcohol Consumption')
plt.ylabel('Grasas Saturadas')
plt.title('Scatter Plot of Grasas Saturadas vs. Alcohol Consumption Colored by Sex')
plt.legend(title='Sex')
plt.grid(True)
plt.savefig("img/scatter_alcohol_vs_grasas_sat.png")
