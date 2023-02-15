import unittest

import numpy as np
from src.main import gaussian_elimination


class TestGaussianElimination(unittest.TestCase):

    def test_has_solution_3x4(self):
        matrix = np.array([
            [  9, -3,  1,  21],
            [ 25,  5,  1,  61],
            [  1,  1,  1,   9],
        ])
        self.assertEqual(
            gaussian_elimination(matrix),
            (('2.0', '1.0', '6.0'), True, False)
        )
        self.assertEqual(
            gaussian_elimination(matrix, frac=True),
            (('2', '1', '6'), True, False)
        )

    def test_has_solution_3x4_zero_first(self):
        matrix = np.array([
            [  0,  1,  1,  3],
            [  1,  2,  0,  7],
            [  2,  0, -1,  8],
        ])
        self.assertEqual(
            gaussian_elimination(matrix), 
            (('5.0', '1.0', '2.0'), True, False)
        )
        self.assertEqual(
            gaussian_elimination(matrix, frac=True),
            (('5', '1', '2'), True, False)
        )

    def test_has_solution_4x5(self):
        matrix = np.array([
            [  1,  1,  1,  1, -2 ],
            [ -1,  1, -1,  1, -10],
            [  0, -2,  0,  1,  0 ],
            [  3,  2,  1,  2, -2 ],
        ])
        self.assertEqual(
            gaussian_elimination(matrix), 
            (('3.0', '-2.0', '1.0', '-4.0'), True, False)
        )
        self.assertEqual(
            gaussian_elimination(matrix, frac=True),
            (('3', '-2', '1', '-4'), True, False)
        )

    def test_has_no_solution_3x4(self):
        matrix = np.array([
            [  4,  3, -2, -5],
            [  4,  1, -1, -8],
            [  8,  8, -5, -6],
        ])
        self.assertEqual(
            gaussian_elimination(matrix), 
            ((), False, False)
        )

    def test_has_unlimited_solutions_3x4(self):
        matrix = np.array([
            [  2, -2,  3,  0],
            [  1, -2,  4, -6],
            [  3, -4,  7, -6],
        ])
        self.assertEqual(
            gaussian_elimination(matrix), 
            (('1.0*t + 6.0', '2.5*t + 6.0', 't'), True, True)
        )
        self.assertEqual(
            gaussian_elimination(matrix, frac=True),
            (('1*t + 6', '(5/2)*t + 6', 't'), True, True)
        )

    def test_has_solution_2x4(self):
        matrix = np.array([
            [ 1, 1, 1, 3],
            [ 1, 2, 3, 6],
        ])
        self.assertEqual(
            gaussian_elimination(matrix),
            (('1.0*t', '3.0 - 2.0*t', 't'), True, True)
        )
        self.assertEqual(
            gaussian_elimination(matrix, frac=True),
            (('1*t', '3 - 2*t', 't'), True, True)
        )

    def test_has_solution_4x4(self):
        matrix = np.array([
            [  1,  1,  1, 15],
            [  2, -1,  7, 50],
            [  3, 11, -9,  1],
            [  1, -1,  1,  5],
        ])
        self.assertEqual(
            gaussian_elimination(matrix),
            (('3.0', '5.0', '7.0'), True, False)
        )
        self.assertEqual(
            gaussian_elimination(matrix, frac=True),
            (('3', '5', '7'), True, False)
        )

    def test_has_no_solution_4x4(self):
        matrix = np.array([
            [  1,  0,  1,  2],
            [  0,  1,  1,  4],
            [  1,  1,  0,  5],
            [  1,  1,  1,  0],
        ])
        self.assertEqual(
            gaussian_elimination(matrix),
            ((), False, False)
        )

    def test_error_first_col_zeros(self):
        matrix = np.array([
            [  0, -3,  1,  21],
            [  0,  5,  1,  61],
            [  0,  1,  1,   9],
        ])
        with self.assertRaises(ValueError):
            gaussian_elimination(matrix)


if __name__ == '__main__':
    unittest.main()