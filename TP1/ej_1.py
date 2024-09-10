import pandas as pd

NUMBER_OF_CLASSES = 2


def calcular_probabilidades_condicionales(df, total_personas):
    probabilidades = {}
    k = NUMBER_OF_CLASSES
    for columna in df.columns:
        if columna != "Nacionalidad":
            prob_1 = (df[columna].sum() + 1) / (total_personas + k)
            probabilidades[columna] = prob_1
    return probabilidades


def clasificar(vector, prob_cond_ingleses, prob_cond_escoceses, prior_ingleses, prior_escoces):
    prob_ingleses = prior_ingleses
    prob_escoces = prior_escoces

    for columna, valor in vector.items():
        if valor == 1:
            prob_ingleses *= prob_cond_ingleses[columna]
            prob_escoces *= prob_cond_escoceses[columna]
        else:
            prob_ingleses *= (1 - prob_cond_ingleses[columna])
            prob_escoces *= (1 - prob_cond_escoceses[columna])

    # Determinar la clase con mayor probabilidad
    if prob_ingleses > prob_escoces:
        return "Inglés"
    else:
        return "Escocés"


def main():
    df = pd.read_excel('data/PreferenciasBritanicos.xlsx')
    # print(df)

    ingleses = df[df["Nacionalidad"] == "I"]
    escoceses = df[df["Nacionalidad"] == "E"]
    # print(ingleses)
    # print(escoceses)

    total_ingleses = len(ingleses)
    total_escoceses = len(escoceses)
    total_personas = total_ingleses + total_escoceses
    #print("Total ingleses", total_ingleses)
    #print("Total escoceses", total_escoceses)
    #print("Total personas", total_personas)

    prior_ingleses = total_ingleses / total_personas
    prior_escoces = total_escoceses / total_personas

    prob_cond_ingleses = calcular_probabilidades_condicionales(
        ingleses, total_ingleses)
    prob_cond_escoceses = calcular_probabilidades_condicionales(
        escoceses, total_escoceses)
    print("P_cond ingleses", prob_cond_ingleses)
    print("P_cond escoceses", prob_cond_escoceses)

    x_1 = {
        "scones": 1,
        "cerveza": 0,
        "wiskey": 1,
        "avena": 1,
        "futbol": 0
    }
    resultado = clasificar(x_1, prob_cond_ingleses,
                           prob_cond_escoceses, prior_ingleses, prior_escoces)
    print(f"El vector {x_1} es clasificado como: {resultado}")

    x_2 = {
        "scones": 0,
        "cerveza": 1,
        "wiskey": 1,
        "avena": 0,
        "futbol": 1
    }
    resultado = clasificar(x_2, prob_cond_ingleses,
                           prob_cond_escoceses, prior_ingleses, prior_escoces)
    print(f"El vector {x_2} es clasificado como: {resultado}")


if __name__ == '__main__':
    main()
