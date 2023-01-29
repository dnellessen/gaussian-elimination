import unittest

import numpy as np
from src.matrix import *


class TestMatrixOperations(unittest.TestCase):
    matrix = np.array([
        [  0,  1,  1,  3 ],
        [  1,  2,  0,  7 ],
        [  2,  0, -1,  8 ],
    ])

    def test_get_body(self):
        self.assertEqual(
            get_body(self.matrix).tolist(), 
            np.array([
                [  0,  1,  1 ],
                [  1,  2,  0 ],
                [  2,  0, -1 ],
            ]).tolist()
        )

    def test_get_tail(self):
        self.assertEqual(
            get_tail(self.matrix).tolist(), 
            np.array([3, 7, 8]).tolist()
        )

    def test_get_cols(self):
        self.assertEqual(
            get_cols(self.matrix).tolist(), 
            np.array([
                [  0,  1,  2 ],
                [  1,  2,  0 ],
                [  1,  0, -1 ],
                [  3,  7,  8 ],
            ]).tolist()
        )

    def test_get_diag(self):
        self.assertEqual(
            get_diag(self.matrix).tolist(), 
            np.array([0, 2, -1]).tolist()
        )

    def test_switch_rows(self):
        self.assertEqual(
            switch_rows(self.matrix, 0, 2).tolist(), 
            np.array([
                [  2,  0, -1,  8 ],
                [  1,  2,  0,  7 ],
                [  0,  1,  1,  3 ],
            ]).tolist()
        )


if __name__ == '__main__':
    unittest.main()
