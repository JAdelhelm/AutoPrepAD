import re
from collections import Counter

def count_big_letters(x):
    if str(x).lower() == "nan":
        return 0
    else:
        return len(re.findall("[A-Z]", str(x)))

def count_small_letters(x):
    if str(x).lower() == "nan":
        return 0
    else:
        return len(re.findall("[a-z]", str(x)))
    
def len_string(x):
    if str(x).lower() == "nan":
        return 0
    else:
        return len(str(x))
    
def count_numbers(x):
    return len(re.findall("[0-9]", str(x)))

def count_operator(x):
    return len(re.findall("[+:=|&<>^*-]", str(x)))

def count_special_char(x):
    return len(re.findall("[$,;?@#'.()%!]", str(x)))

def count_unqiue_char(x):
    return len(Counter(str(x)))