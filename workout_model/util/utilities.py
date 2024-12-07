import re
import numpy as np

# Custom sorting function to support the sorting of a list that contains both numbers and strings. Abides by
# standard "file system" sorting, where numbers are early and alphas are late.
def fileStandardSortKey(item):
    # Check if item is a number (int or float), numbers come first
    if(isinstance(item, (int, float, np.integer, np.floating))):
        return (0, item)  # Numbers are prioritized with category 0
    elif(isinstance(item, str)):
        return (1, item.lower())  # Strings follow numbers with category 1
    else:
        raise TypeError(f"Unsupported type: {type(item)}")

# This function removes a set of preconfigured "bad characters" from a string.
def cleanBadCharacters(thisString):
    if(isinstance(thisString, str)):
        # Define a blacklist of unwanted characters (expand as needed)
        blacklist = r'[\u200b\u200c\u200d\u202a-\u202e]'  # Zero-width spaces, bidirectional formatting chars
        return re.sub(blacklist, "", thisString).strip()  # Remove blacklisted chars and strip whitespace
    else:
        return thisString