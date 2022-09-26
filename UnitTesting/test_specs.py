# test_specs.py
"""Python Essentials: Unit Testing.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/22/22
"""

from multiprocessing.sharedctypes import Value
import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    assert specs.smallest_factor(8) == 2, "failed"
    assert specs.smallest_factor(5) == 5, "failed"
    assert specs.smallest_factor(33) == 3

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    assert specs.month_length("September") == 30, "failed on months with 30 days"
    assert specs.month_length("July") == 31, "failed on months with 31 days"
    assert specs.month_length("February", True) == 29, "failed on leap year"
    assert specs.month_length("February") == 28, "failed on not a leap year"
    assert specs.month_length("not a month") == None, "failed on not a month"

# Problem 3: write a unit test for specs.operate().
def test_operate():
    
    assert specs.operate(1, 2, "+") == 3, "failed on add"
    assert specs.operate(2, 1, "-") == 1, "failed on minus"
    assert specs.operate(2, 3, "*") == 6, "failed on multiply"
    assert specs.operate(6, 3, "/") == 2, "failed on divide"
    with pytest.raises(TypeError) as excinfo:
        specs.operate(1, 2, 1) 
    assert excinfo.value.args[0] == "oper must be a string"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(4, 0, "/")
    assert excinfo.value.args[0] == "division by zero is undefined"
    with pytest.raises(ValueError) as excinfo:
        specs.operate(1, 2, "=")
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"


# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)


# Problem 5: Write test cases for Set.
