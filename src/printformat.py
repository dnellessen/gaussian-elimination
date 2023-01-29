
def printf(L: tuple, has_solution: bool, is_unlimited: bool):
    '''
    Prints formatted solution set.

    Parameters
    ----------
    L: tuple
        Solution set
    has_solution: bool
    is_unlimited: bool
    '''

    if not has_solution:
        print(r"L = {}")
    elif not is_unlimited:
        print("L = {(", '; '.join(L), ")}")
    else:
        print("L = {(", '; '.join(L), ") | t∈R }")
