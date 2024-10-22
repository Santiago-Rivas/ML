#!/bin/bash

# python -u python/ej_2_poly.py output/poly_and_rbf/ poly "not final" > logs

# python -u python/ej_2_poly.py output/rbf_all/ rbf "final" > logs

python -u python/ej_2_poly.py output/poly_and_rbf/ linear "not final" > logs

# python -u python/ej_2_poly.py output/poly_and_rbf/ sigmoid "no final" > logs
#
# c = 100 d 1, coef 0, gamm 10      YES
# c = 100 d 1, coef 0, gamm scale   YES
#
# c = 100 d 1, coef 10, gamm 10     YES
# c = 100 d 1, coef 10, gamm scale  YES
#
# c = 100 d 1, coef 100, gamm 10    YES
# c = 100 d 1, coef 100, gamm scale YES
#
# c = 100 d 2, coef 0, gamm scale   YES
# c = 100 d 2, coef 1, gamm scale   YES
# c = 100 d 2, coef 10, gamm scale  YES
# c = 100 d 2, coef 100, gamm scale YES
#
# c = 100 d 3, coef 0, gamm scale   YES
# c = 100 d 3, coef 1, gamm scale   YES
# c = 100 d 3, coef 10, gamm scale  YES
#
# c = 10 d 6, coef 10, gamm 10
# c = 100 d 6, coef 100, gamm 1
# c = 100 d 6, coef 100, gamm 10
# c = 100 d 6, coef 100, gamm scale
