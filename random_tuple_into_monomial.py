import random
import sympy
def generate_tuples(tuple_length, coeff_range, deg_range, amount):
    tuples= []
    for i in range(amount):
        coeff= random.randint(*coeff_range)
        deg= [random.randint(*deg_range) for _ in range(tuple_length - 1)]
        tuples.append((coeff, *deg))
    return tuples

x= generate_tuples(6, (1,10), (0,10), 1000)
print(x)

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

print(tuple_monomial_list(x))

