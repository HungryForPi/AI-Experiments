import random
import sympy
from sympy.polys.numberfields.utilities import coeff_search


def generate_tuples(tuple_length, coeff_range, deg_range, amount):
    tuples= []
    for i in range(amount):
        coeff= random.randint(*coeff_range)
        deg= [random.randint(*deg_range) for _ in range(tuple_length - 1)]
        tuples.append((coeff, *deg))
    return tuples

def tuple_monomial_transformer(tuple):
    coef= tuple[0]
    exponents = tuple[1:]
    variables= sympy.symbols(f'x_1:{len(exponents)+1}')

    monomial= coef
    for var, exp in zip(variables,exponents):
        monomial *=  var**exp
    return monomial

def tuple_monomial_list(tuple_list):
    return [tuple_monomial_transformer(t) for t in tuple_list]

def group_tuple(tuple_list, num_groups):
    group_size= len(tuple_list) // num_groups
    return [tuple_list[i*group_size:(i+1)*group_size] for i in range(num_groups)]

def grouped_tuples_to_polynomials(grouped_tuples):
    """
    Convert groups of tuples into polynomials.
    """
    polys = []
    for group in grouped_tuples:
        monomials = [tuple_monomial_transformer(t) for t in group]
        polys.append(sum(monomials))
    return polys

tuples=generate_tuples(tuple_length=6,
                       coeff_range=(-10,11),
                       deg_range=(0,11),
                       amount=1800)

grouped_tuples=group_tuple(tuple_list=tuples,
                           num_groups=180)

polynomials= grouped_tuples_to_polynomials(grouped_tuples)

for i in range(len(grouped_tuples)):
    print(f"Group {i+1}: {grouped_tuples[i]}")
    print(f"Polynomial {i+1}: {polynomials[i]}\n")





