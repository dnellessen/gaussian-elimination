[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

# Gaussian Elimination

Perform the Gaussian elimination for any n x n+1 matrix and get a solution set for one, no, or unlimited solutions.


## Exapmles
(1)

$$ \left(\begin{matrix}
1 & 1 & 1 & 1 & | & -2\\
-1 & 1 & -1 & 1 & | & -10\\
0 & -2 & 0 & 1 & | & 0\\
3 & 2 & 1 & 2 & | & -2\\
\end{matrix}\right) $$

$$
L = {\\{( 3; -2; 1; -4 )}\\}
$$

(2)

$$ \left(\begin{matrix}
4 & 3 & -2 & | & -5\\
4 & 1 & -1 & | & -8\\
8 & 8 & -5 & | & -6
\end{matrix}\right) $$

$$
L = {\\{}\\}
$$

(3)

$$ \left(\begin{matrix}
2 & -2 & 3 & | & 0\\
1 & -2 & 4 & | & -6\\
3 & -4 & 7 & | & -6
\end{matrix}\right) $$

$$
L = {\\{( 2 - t; 2t + 2; t ) | t\in\mathbb{R}}\\}
$$

## Getting Started

### Installing

```bash
$ git clone https://github.com/dnellessen/gaussian-elimination.git
```
or download ZIP.

### Dependencies
* numpy
* sympy
```bash
$ pip install -r requirements.txt
```


### Executing program

Let's say our linear system of equations is:

$$ 6x  + 2y - 1z = 5\ $$

$$ 3x - 4y  - 2z = 16\ $$

$$ 2x  - 1y - 2z = 5 $$

In matrix form we get:

$$ \left(\begin{matrix}
6 & 2 & -1 & | & 5\\
3 & -4 & -2 & | & 16\\
2 & -1 & -2 & | & 5
\end{matrix}\right) $$

To solve the system go into the according directory and run `main.py`.
```bash
$ cd gaussian-elimination
$ python3 src/main.py
```

Now you can enter each row, seperating the values with a space.\
When done, enter a new line and the system will be solved.
```
$ python3 src/main.py
6 2 -1 5
3 -4 -2 16
2 -1 -2 5

L = {( 2.0; -3.0; 1.0 )}
```

If you use the argument `-f` the solution set will be computed using fractions.
