import numpy as np


def get_body(matrix: np.ndarray):
    '''
    Returns given matrix without last column.

    Parameter
    ---------
    matrix: np.ndarray

    Returns
    -------
    matrix: np.ndarray
    '''

    return matrix[:, :-1]


def get_tail(matrix: np.ndarray):
    '''
    Returns the last column of given matrix.

    Parameter
    ---------
    matrix: np.ndarray

    Returns
    -------
    matrix: np.ndarray
    '''

    return matrix[:, -1]


def get_cols(matrix: np.ndarray):
    '''
    Returns columns of given matrix.

    Parameter
    ---------
    matrix: np.ndarray

    Returns
    -------
    matrix: np.ndarray
    '''

    return matrix.T


def get_diag(matrix: np.ndarray):
    '''
    Returns diagonal of matrix.

    Parameter
    ---------
    matrix: np.ndarray

    Returns
    -------
    matrix: np.ndarray
    '''

    return matrix.diagonal()


def switch_rows(matrix: np.ndarray, m: int, n: int):
    '''
    Switches rows of matrix.

    Parameters
    ----------
    matrix: np.ndarray
    m: int
        index of first row
    n: int
        index of second row

    Returns
    -------
    matrix: np.ndarray
    '''

    matrix[[m, n], :] = matrix[[n, m], :]
    return matrix
