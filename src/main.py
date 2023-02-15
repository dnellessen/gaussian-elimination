import numpy as np
from sympy import simplify

from fractions import Fraction
import re
import sys

if __name__ == '__main__':
    from matrix import *
    from printformat import printf
else:
    from .matrix import *
    from .printformat import printf


def convert_to_fraction(L: tuple):
    '''
    Converts floats to fractions in solution set.

    Parameters
    ----------
    L: tuple[str]

    Returns
    ----------
    L_as_frac: tuple[str]
    '''

    L_as_frac = []

    for i, sol in enumerate(L):
        sol_as_frac = sol

        floats = re.findall(r"[-+]?(?:\d*\.*\d+)", sol)     # isolate floats (useful with unlimited solution)
        for as_float in floats:
            as_fraction = str(Fraction(as_float).limit_denominator(1000))

            if float(eval(as_fraction)).is_integer():   # ignore .0
                sol_as_frac = sol_as_frac.replace(as_float, as_fraction)
            else:
                sol_as_frac = sol_as_frac.replace(as_float, f'({as_fraction})')

        L_as_frac.append(sol_as_frac)

    return tuple(L_as_frac)


def format_unlimited(matrix: np.ndarray, frac: bool = False) -> tuple[str]:
    '''
    Returns solution set of given matrix for unlimited solutions (Last row empty).

    Parameter
    ----------
    matrix: np.ndarray
    frac: bool (Default: False)
        Solution set with franctions instead of decimals

    Returns
    -------
    L: tuple[str]
        Solution set
    '''

    matrix_rows, matrix_cols = matrix.shape

    L = [0] * (matrix_cols-1)   # fill solution set with zeros
    L[-1] = 't'

    col = len(L) - 2    # seconds last column
    matrix_body = get_body(matrix)
    matrix_tail = get_tail(matrix)  # one dimensional

    for row in matrix_body[:-1][::-1]:  # for every row ignoring last and backwards
        numerator = []
        if matrix_tail[col] != 0:   # add tail element if not zero
            numerator.append(str(matrix[col, -1]))

        for n in range(matrix_cols-2, col, -1):   # run until unknown variable is reached
            symbol = '+' if row[n] < 0 else '-'
            numerator.append(f"{symbol}{abs(row[n])}*({L[n]})")

        L[col] = str(simplify(f"({''.join(numerator)}) / {row[col]}"))

        col -= 1

    L = tuple(L) if not frac else convert_to_fraction(tuple(L))
    return L


def test_rest_equations(L: tuple[str], rest_equations: np.ndarray) -> bool:
    '''
    Checks if the rest equations are result to true expressions
    when plugginf the solution set in.

    Parameter
    ----------
    L: tuple[str]
    rest_equations: np.ndarray

    Returns
    -------
    success: bool
    '''

    t = 1   # for unlimited solutions

    for eq in rest_equations:
        res = 0
        for i in range(len(L)):
            res += eval(f"({L[i]})*({eq[i]})")
        if round(res, 10) != eq[-1]:
            return False

    return True


def gaussian_elimination(matrix: np.ndarray, frac: bool = False) -> tuple[tuple[str], bool, bool]:
    '''
    Solves a linear system of equations using the Gaussian elimination.
    Returns a solution set for one, no, or unlimited solutions.

    Parameter
    ----------
    matrix: np.ndarray
    frac: bool (Default: False)
        Solution set with franctions instead of decimals

    Returns
    ----------
    L: tuple[str]
        Solution set
    has_solution: bool
    is_unlimited: bool
    
    Raises
    ------
    ValueError
        If first column is filled with zeros.

    Examples
    -------
    >>> martix = np.array([
    ...    [  9, -3,  1,  21],
    ...    [ 25,  5,  1,  61],
    ...    [  1,  1,  1,   9],
    ... ])
    >>> 
    >>> gaussian_elimination(matrix)
    ('2.0', '1.0', '6.0'), True, False
    >>> gaussian_elimination(matrix, frac=True)
    ('2', '1', '6'), True, False
    >>> 
    >>> 
    >>> matrix = np.array([
    ...    [  1,  1,  1,  1, -2 ],
    ...    [ -1,  1, -1,  1, -10],
    ...    [  0, -2,  0,  1,  0 ],
    ...    [  3,  2,  1,  2, -2 ],
    ... ])
    >>> 
    >>> gaussian_elimination(matrix)
    ('3.0', '-2.0', '1.0', '-4.0'), True, False
    >>> gaussian_elimination(matrix, frac=True)
    ('3', '-2', '1', '-4'), True, False
    >>> 
    >>> 
    >>> matrix = np.array([
    ...    [  4,  3, -2, -5],
    ...    [  4,  1, -1, -8],
    ...    [  8,  8, -5, -6],
    ... ])
    >>> 
    >>> gaussian_elimination(matrix)
    (), False, False
    >>> 
    >>> 
    >>> matrix = np.array([
    ...    [  2, -2,  3,  0],
    ...    [  1, -2,  4, -6],
    ...    [  3, -4,  7, -6],
    ... ])
    >>> 
    >>> gaussian_elimination(matrix)
    ('1.0*t + 6.0', '2.5*t + 6.0', 't'), True, True
    >>> gaussian_elimination(matrix, frac=True)
    ('1*t + 6', '(5/2)*t + 6', 't'), True, True
    >>> 
    >>> 
    >>> matrix = np.array([
    ...    [ 1, 1, 1, 3],
    ...    [ 1, 2, 3, 6],
    ... ])
    >>> 
    >>> gaussian_elimination(matrix)
    ('1.0*t', '3.0 - 2.0*t', 't'), True, True
    >>> gaussian_elimination(matrix, frac=True)
    ('1*t', '3 - 2*t', 't'), True, True
    >>> 
    >>> 
    >>> matrix = np.array([
    ...    [  1,  1,  1, 15],
    ...    [  2, -1,  7, 50],
    ...    [  3, 11, -9,  1],
    ...    [  1, -1,  1,  5],
    ... ])
    >>> 
    >>> gaussian_elimination(matrix)
    ('3.0', '5.0', '7.0'), True, False
    >>> gaussian_elimination(matrix, frac=True)
    ('3', '5', '7'), True, False
    >>> 
    >>> 
    >>> matrix = np.array([
    ...    [  1,  0,  1,  2],
    ...    [  0,  1,  1,  4],
    ...    [  1,  1,  0,  5],
    ...    [  1,  1,  1,  0],
    ... ])
    >>> 
    >>> gaussian_elimination(matrix)
    (), False, False
    >>> 
    >>> 
    >>> martix = np.array([
    ...    [  0, -3,  1,  21],
    ...    [  0,  5,  1,  61],
    ...    [  0,  1,  1,   9],
    ... ])
    >>> 
    >>> gaussian_elimination(matrix)
    ValueError: entire first row cannot be 0
    >>> 
    '''

    matrix = matrix.astype(np.float64)
    matrix_rows, matrix_cols = matrix.shape

    rest_equations = []

    ## check matrix shape and expand if necessary ##
    num_vabiables = matrix_cols - 1
    if num_vabiables > matrix_rows:     # more variables than equations
        diff = num_vabiables - matrix_rows
        for _ in range(diff):
            matrix = np.vstack([matrix, np.zeros(matrix_cols)])
    elif num_vabiables < matrix_rows:   # less variables than equations
        diff = matrix_rows - num_vabiables
        for _ in range(diff):
            rest_equations.append(matrix[-1])
            matrix = matrix[:-1]

    rest_equations = np.array(rest_equations)
    matrix_rows, matrix_cols = matrix.shape

    ## check matrix shape and expand if necessary ##
    while matrix_cols != matrix_rows + 1:
        matrix = np.vstack([matrix, np.zeros(matrix_cols)])
        matrix_rows, matrix_cols = matrix.shape

    ## switch rows since the first element cannot be 0 ##
    i = 1
    while matrix[0][0] == 0:
        if i == matrix_rows: 
            raise ValueError("entire first row cannot be 0")
        matrix = switch_rows(matrix, 0, -i)
        i += 1

    ## empty triangle ##
    for i in range(matrix_rows - 1):    # for every row excluding last(index)
        row0 = matrix[i]    # next row (starting with first)
        val0 = row0[i]      # first value of row (i not 0 since triangle)
        if val0 == 0:
            continue
        for row_index, row in enumerate(matrix[i+1:], i+1):     # for every following row (w\ index)
            val1 = row[i]   # first value of row (i not zero since triangle)
            if val1 != 0:
                mul_row0 = row0 * val1
                new_row  = row  * val0 - mul_row0   
                matrix[row_index] = new_row

    ## rearrange matrix (triangle) ##
    matrix0 = matrix.copy()
    for row in matrix:
        num_of_zeros = 0
        for col in row:
            if col == 0: num_of_zeros += 1
            else: break
        try:
            matrix0[num_of_zeros] = row
        except IndexError:
            matrix0[-1] = row
    matrix = matrix0

    ## check if system has a solution ##
    for row in matrix:
        if not any(row[:-1]):   # if last row is filled with zeros
            # no solution
            if row[-1] != 0:
                has_solution, is_unlimited = False, False
                return (), has_solution, is_unlimited
            # unlimited solutions
            else:
                has_solution, is_unlimited = True, True
                L = format_unlimited(matrix, frac)

                if rest_equations.any():
                    if not test_rest_equations(L, rest_equations):
                        has_solution, is_unlimited = False, False
                        return (), has_solution, is_unlimited

                return L, has_solution, is_unlimited

    ## get solution set ##
    L = np.ones(matrix_cols-1)     # solution set filled with ones
    max_i = matrix_rows-1          # ignore last row

    for i in range(max_i, -1, -1):   # for every row (backwards index)
        col = matrix_cols - 2        # seconds last column
        for _ in range(max_i - i, 0, -1):       # run until unknown variable is reached
            matrix[i, col]  *=  L[col]          # multiply body with calculated variable
            matrix[i, -1] += -matrix[i, col]    # update last element
            matrix[i, col]  -=  matrix[i, col]  # update body element
            col -= 1

        # spimplify to have diagonal ones
        L[i] = matrix[i, -1] / matrix[i, i]
        matrix[i, i] /= matrix[i, i]
        matrix[i, -1] = L[i]

    has_solution, is_unlimited = True, False
    L = tuple(L.astype(str)) if not frac else convert_to_fraction(L.astype(str))

    if rest_equations.any():
        if not test_rest_equations(L, rest_equations):
            has_solution, is_unlimited = False, False
            return (), has_solution, is_unlimited

    return L, has_solution, is_unlimited


if __name__ == "__main__":
    print("Enter matrix:")

    matrix = []
    while True:
        n = input()
        if not n: break
        matrix.append([eval(i) for i in n.split(' ')])

    matrix = np.array(matrix)
    frac = '-f' in sys.argv

    printf(*gaussian_elimination(matrix, frac))
