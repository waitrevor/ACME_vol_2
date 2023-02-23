# regular_expressions.py
"""Volume 3: Regular Expressions.
<Name> Trevor Wai
<Class> Section 1
<Date> 2/22/23
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #A compiled regular expression pattern string 'python'
    return re.compile(r'python')

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #A compiled regular expression pattern string
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #A compiled regular expression pattern string
    return re.compile(r'^(Book|Mattress|Grocery) (store|supplier)$')

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #Compiled regular expression pattern string
    return re.compile(r"^[a-zA-Z_][\w_]* *(= *(\d+(\.\d+)?|'[^']*'|[a-zA-Z_][\w_]*))?$")

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    #Puts colons behind else finally and try
    pat1 = re.compile(r"\b(else|finally|try)", re.MULTILINE)
    code = pat1.sub(r"\1:",code)

    #Puts colons behind if elif for while except with def and class
    pat2 = re.compile(r"\b(if|elif|for|while|except|with|def|class)(.*)", re.MULTILINE)
    code = pat2.sub(r"\1\2:",code)

    return code

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """

    contacts = dict()

    with open(filename) as infile:
        lines = infile.readlines()

    #Regular Expression for Names Birthdays emails and phone numbers
    nAMES = re.compile(r"([A-Z][a-zA-Z]* \w?.? ?[A-Z][a-zA-Z]*)")
    bIRTHDAY = re.compile(r"(\d{1,2})\/(\d{1,2})\/(\d{2,4})")
    eMAIL = re.compile(r"[\w.]*@[\w]*.[\w.]*[\w]*")
    pHONE = re.compile(r"1?-?\(?(\d{3})\)?-?(\d{3}-\d{4})")

    for line in lines:
        name = nAMES.findall(line)[0]
        #Saves each contacts name into the dictionar
        contacts[name] = dict()

        birthday = bIRTHDAY.findall(line)
        email = eMAIL.findall(line)
        phone = pHONE.findall(line)

        #Saves birthdays
        if birthday == []:
            contacts[name]['birthday'] = None
        else:
            birthday = list(birthday[0])
            #Formats the birthdays
            if len(birthday[0]) == 1:
                birthday[0] = '0' + birthday[0]
            if len(birthday[1]) == 1:
                birthday[1] = '0' + birthday[1]
            if len(birthday[2]) == 2:
                birthday[2] = '20' + birthday[2]
            contacts[name]['birthday'] = birthday[0] + '/' + birthday[1] + '/' + birthday[2]

        #Saves Emails
        if email == []:
            contacts[name]['email'] = None
        else:
            contacts[name]['email'] = email[0]

        #Saves phone numbers
        if phone == []:
            contacts[name]['phone'] = None
        else:
            contacts[name]['phone'] = '(' + phone[0][0] + ')' + phone[0][1]

    return contacts
