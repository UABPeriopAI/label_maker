import re


# Function to normalize text
def normalize_text(s):
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\. ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s
