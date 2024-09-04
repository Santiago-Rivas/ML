# Probabilidades iniciales
P_E = 0.10  # Probabilidad de ser estudiante
P_G = 0.90  # Probabilidad de ser graduado

# Probabilidades condicionales dado que es estudiante
P_A_given_E = 0.95
P_B_given_E = 0.05
P_C_given_E = 0.02
P_D_given_E = 0.20

# Probabilidades condicionales dado que es graduado
P_A_given_G = 0.03
P_B_given_G = 0.82
P_C_given_G = 0.34
P_D_given_G = 0.92

# Probabilidad de que le guste A y C, pero no le gusten B y D
P_A_C_not_B_not_D_given_E = P_A_given_E * \
    P_C_given_E * (1 - P_B_given_E) * (1 - P_D_given_E)
P_A_C_not_B_not_D_given_G = P_A_given_G * \
    P_C_given_G * (1 - P_B_given_G) * (1 - P_D_given_G)

# Probabilidad total de escuchar A y C,
# pero no B y D (Teorema de la probabilidad total)
P_A_C_not_B_not_D = P_A_C_not_B_not_D_given_E * \
    P_E + P_A_C_not_B_not_D_given_G * P_G

# Probabilidad de que sea estudiante dado que le gustan A y C,
# pero no B y D (Teorema de Bayes)
P_E_given_A_C_not_B_not_D = (
    P_A_C_not_B_not_D_given_E * P_E) / P_A_C_not_B_not_D

# Probabilidad de que sea graduado dado que le gustan A y C,
# pero no B y D (Teorema de Bayes)
P_G_given_A_C_not_B_not_D = (
    P_A_C_not_B_not_D_given_G * P_G) / P_A_C_not_B_not_D

# Imprimir resultados
print(f"Probabilidad de que sea estudiante: {P_E_given_A_C_not_B_not_D:.4f}")
print(f"Probabilidad de que sea graduado: {P_G_given_A_C_not_B_not_D:.4f}")
