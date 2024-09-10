import pandas as pd
import numpy as np


def get_ranking_distribution(column):
    ranking = np.array([0, 0, 0, 0])
    for val in column:
        if val == 1:
            ranking[0] += 1
        if val == 2:
            ranking[1] += 1
        if val == 3:
            ranking[2] += 1
        if val == 4:
            ranking[3] += 1
    return ranking / len(column)

def get_gre_distribution(rank_column, gre_column):
    ranking = np.array([0, 0, 0, 0])
    totals = np.array([0, 0, 0, 0])
    for i in range(len(rank_column)):
        if rank_column[i] == 1:
            if gre_column[i] >= 500:
                ranking[0] += 1
            totals[0] += 1
        if rank_column[i] == 2:
            if gre_column[i] >= 500:
                ranking[1] += 1
            totals[1] += 1
        if rank_column[i] == 3:
            if gre_column[i] >= 500:
                ranking[2] += 1
            totals[2] += 1
        if rank_column[i] == 4:
            if gre_column[i] >= 500:
                ranking[3] += 1
            totals[3] += 1
    return ranking / totals

def get_gpa_distribution(rank_column, gpa_column):
    ranking = np.array([0, 0, 0, 0])
    totals = np.array([0, 0, 0, 0])
    for i in range(len(rank_column)):
        if rank_column[i] == 1:
            if gpa_column[i] >= 3:
                ranking[0] += 1
            totals[0] += 1
        if rank_column[i] == 2:
            if gpa_column[i] >= 3:
                ranking[1] += 1
            totals[1] += 1
        if rank_column[i] == 3:
            if gpa_column[i] >= 3:
                ranking[2] += 1
            totals[2] += 1
        if rank_column[i] == 4:
            if gpa_column[i] >= 3:
                ranking[3] += 1
            totals[3] += 1
    return ranking / totals

def get_admit_distribution(rank_column, gre_column, gpa_column, admit_column):
    
    admit_distribution = np.zeros((4, 4))
    totals_distribution = np.zeros((4, 4))
    
    for i in range(len(rank_column)):
        rank = rank_column[i]
        if gre_column[i] >= 500 and gpa_column[i] >= 3: 
            col = 0
        elif gre_column[i] < 500 and gpa_column[i] >= 3:
            col = 1
        elif gre_column[i] >= 500 and gpa_column[i] < 3:
            col = 2
        elif gre_column[i] < 500 and gpa_column[i] < 3:
            col = 3
        if admit_column[i] == 1:
                admit_distribution[rank-1][col] += 1
        totals_distribution[rank-1][col] += 1

    return admit_distribution /totals_distribution
    
def get_admit_prob(admit_column):
    count = 0
    for val in admit_column:
        if(val == 1):
            count += 1
    return count / len(admit_column)

df = pd.read_csv(r'TP1\data\binary.csv')

admit_column = df['admit']
gre_column = df['gre']
gpa_column = df['gpa']
rank_column = df['rank']

#admit_prob = get_admit_prob(admit_column)
admit_distribution = get_admit_distribution(rank_column, gre_column, gpa_column, admit_column)
gre_distribution = get_gre_distribution(rank_column, gre_column)
gpa_distribution = get_gpa_distribution(rank_column, gpa_column)
ranking_distribution = get_ranking_distribution(rank_column)

print(admit_distribution)
print(gre_distribution)
print(gpa_distribution)

no_admit_given_rank1 = 0

no_admit_given_rank1 += ((1-admit_distribution[0][0])*(gre_distribution[0])*(gpa_distribution[0]))
no_admit_given_rank1 += ((1-admit_distribution[0][1])*(1-gre_distribution[0])*(gpa_distribution[0]))
no_admit_given_rank1 += ((1-admit_distribution[0][2])*(gre_distribution[0])*(1-gpa_distribution[0]))
no_admit_given_rank1 += ((1-admit_distribution[0][3])*(1-gre_distribution[0])*(1-gpa_distribution[0]))


print(no_admit_given_rank1)

admit_given_rank2 = admit_distribution[1][1]

cant_admit = 0

for linea in admit_column:
    cant_admit += linea

prob_admit = cant_admit / len(admit_column)
prob_no_admit = 1 - prob_admit


