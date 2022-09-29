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
    """Tests the various cases of the add function"""
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8
    #Testing for the add magic method in the Fraction Class
    assert specs.Fraction(3, 1) + specs.Fraction(4,2) == specs.Fraction(5, 1)

def test_divide():
    """Tests the various cases of the divide function"""
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    #Raises an error if the denom input is 0
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    """Tests the various cases of the smallest_factor function"""
    assert specs.smallest_factor(8) == 2, "failed"
    assert specs.smallest_factor(5) == 5, "failed"
    assert specs.smallest_factor(33) == 3

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    """Tests the various cases of the moth_length function"""
    assert specs.month_length("September") == 30, "failed on months with 30 days"
    assert specs.month_length("July") == 31, "failed on months with 31 days"
    assert specs.month_length("February", True) == 29, "failed on leap year"
    assert specs.month_length("February") == 28, "failed on not a leap year"
    assert specs.month_length("not a month") == None, "failed on not a month"

# Problem 3: write a unit test for specs.operate().
def test_operate():
    """Tests the various cases of the operate function"""
    assert specs.operate(1, 2, "+") == 3, "failed on add"
    assert specs.operate(2, 1, "-") == 1, "failed on minus"
    assert specs.operate(2, 3, "*") == 6, "failed on multiply"
    assert specs.operate(6, 3, "/") == 2, "failed on divide"
    #Raises an error if the operator input is not a string
    with pytest.raises(TypeError) as excinfo:
        specs.operate(1, 2, 1) 
    assert excinfo.value.args[0] == "oper must be a string"
    #Raises an error if there is a division by zero
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(4, 0, "/")
    assert excinfo.value.args[0] == "division by zero is undefined"
    #Raises an error if the incorrect operator is inputed
    with pytest.raises(ValueError) as excinfo:
        specs.operate(1, 2, "=")
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"


# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    #Sets up 3/1 as a fraction
    frac_3_1 = specs.Fraction(3,1)
    return frac_1_3, frac_1_2, frac_n2_3, frac_3_1

def test_fraction_init(set_up_fractions):
    """Tests the various cases of the constructor"""
    frac_1_3, frac_1_2, frac_n2_3, frac_3_1 = set_up_fractions
    #Raises errors if the denom of the fraction is 0
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(1, 0)
    assert excinfo.value.args[0] == "denominator cannot be zero"
    #Raises an error if the numer and denom aren't integers
    with pytest.raises(TypeError) as excinfo:
        specs.Fraction("", "")
    assert excinfo.value.args[0] == "numerator and denominator must be integers"
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
   

def test_fraction_str(set_up_fractions):
    """Tests various cases of the string magic method"""
    frac_1_3, frac_1_2, frac_n2_3, frac_3_1 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(frac_3_1) == "3"

def test_fraction_float(set_up_fractions):
    """Tests the various cases of the float magic method"""
    frac_1_3, frac_1_2, frac_n2_3, frac_3_1 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    """Tests the various cases of the equivalent magic method"""
    frac_1_3, frac_1_2, frac_n2_3, frac_3_1 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert 3.0 == frac_3_1

def test_fraction_sub(set_up_fractions):
    """Tests the various cases of the subtract magic method"""
    frac_1_3, frac_1_2, frac_n2_3, frac_3_1 = set_up_fractions
    assert frac_1_2 - frac_1_3 == specs.Fraction(1, 6)

def test_fraction_mul(set_up_fractions):
    """Tests the various cases of the multiply magic method"""
    frac_1_3, frac_1_2, frac_n2_3, frac_3_1 = set_up_fractions
    assert frac_1_2 * frac_1_3 == specs.Fraction(1, 6)
    assert frac_n2_3 * frac_3_1 == specs.Fraction(-2, 1)

def test_fraction_truediv(set_up_fractions):
    """Tests the various cases of the truediv magic method"""
    frac_1_3, frac_1_2, frac_n2_3, frac_3_1 = set_up_fractions
    assert frac_1_2 / frac_1_3 == specs.Fraction(3, 2)
    #Raises an error if there is a division by zero
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(0, 3) / specs.Fraction(0, 4)
    assert excinfo.value.args[0] == "cannot divide by zero"


# Problem 5: Write test cases for Set.
def test_count_sets():
    """Tests the various cases of the count_sets function"""
    assert specs.count_sets(["1022", "1122", "0100", "2021",
                            "0010", "2201", "2111", "0020",
                            "1102", "0210", "2110", "1020"]) == 6
    #Raises an error if there is not exactly 12 cards
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets([""])
    assert excinfo.value.args[0] == "there are not exactly 12 cards"
    #Raises an error if all the cards are not unique
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1022", "1022", "0100", "2021",
                            "0010", "2201", "2111", "0020",
                            "1102", "0210", "2110", "1020"])
    assert excinfo.value.args[0] == "the cards are not all unique"
    #Raises an error if one or more cards has more than 4 digits
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["10220", "1022", "0100", "2021",
                            "0010", "2201", "2111", "0020",
                            "1102", "0210", "2110", "1020"])
    assert excinfo.value.args[0] == "one or more cards does not have exactly 4 digits"
    #Raises an error if one or more cards has a character other than 0, 1, or 2
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1023", "1022", "0100", "2021",
                            "0010", "2201", "2111", "0020",
                            "1102", "0210", "2110", "1020"])
    assert excinfo.value.args[0] == "one or more cards has a character other than 0, 1, or 2"

def test_is_set():
    """Tests the various cases of the is_set function"""
    assert specs.is_set("1022", "1122", "1020") == False
    assert specs.is_set("0122", "1011", "2200") == True
    assert specs.is_set("1122", "1011", "1200") == True
