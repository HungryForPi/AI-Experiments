import random as rand
import sympy

#GENERATE RANDOM POLY
def gen_ran_poly(variables, max_degree, terms, coeff_range):
    poly = 0
    for _ in range(terms):
        coeff = rand.randint(*coeff_range)
        if coeff == 0:
            continue
        exponents = [rand.randint(0, max_degree) for _ in variables]
        term = coeff * sympy.Mul(*[v**e for v, e in zip(variables, exponents)])
        poly += term
    return sympy.expand(poly)

num_vars = 6
var_symbols = sympy.symbols(f'x1:{num_vars + 1}')
ideal_len = 6     # ideal length
num_ideals = 30   # number of ideals to generate

total_ideals = []
for j in range(num_ideals):
    poly_tuple = tuple(
        gen_ran_poly(
            variables=var_symbols,
            max_degree=rand.randint(1, 10),
            terms=6,
            coeff_range=(-10, 11)
        )
        for _ in range(ideal_len)
    )

    ideal_monomials = []
    for poly in poly_tuple:
        monomials = []
        for term in poly.as_ordered_terms():
            coeff = term.as_coeff_mul(*var_symbols)[0]
            exps = [sympy.degree(term, v) for v in var_symbols]
            monomials.append((coeff, *exps))
        ideal_monomials.append(monomials)

    total_ideals.append(ideal_monomials)

for i, ideal in enumerate(total_ideals, 1):
    print(f"\nIdeal {i}: \n{poly_tuple}")
    for j, poly_mons in enumerate(ideal, 1):
        print(f"({i}) Poly {j}: {poly_mons}")


