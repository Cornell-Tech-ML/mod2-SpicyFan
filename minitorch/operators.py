"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> bool:
    return x < y

def eq(x: float, y: float) -> bool:
    return x == y

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    return abs(x - y) < tol

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    return max(0, x)

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    return d / x

def inv(x: float) -> float:
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    return -d / (x * x)

def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

from typing import Callable, Iterable, List

# map
def map(fn: Callable[[float], float], ls: Iterable[float]) -> List[float]:
    return [fn(x) for x in ls]

# zipWith
def zipWith(fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
    return [fn(x, y) for x, y in zip(ls1, ls2)]

# Reduce
def reduce(fn: Callable[[float, float], float], ls: Iterable[float], start: float) -> float:
    result = start
    for x in ls:
        result = fn(result, x)
    return result

# negList
def negList(ls: Iterable[float]) -> List[float]:
    return map(lambda x: -x, ls)

# addList
def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
    return zipWith(lambda x, y: x + y, ls1, ls2)

# Sum
def sum(ls: Iterable[float]) -> float:
    return reduce(lambda x, y: x + y, ls, 0.0)

# prod
def prod(ls: Iterable[float]) -> float:
    return reduce(lambda x, y: x * y, ls, 1.0)

def prod(x):
    result = 1
    for val in x:
        result *= val
    return result