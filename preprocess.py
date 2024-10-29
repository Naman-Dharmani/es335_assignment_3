import re

# file_name = "data/test"
file_name = "data/sherlock_holmes"
file_ext = ".txt"

with open(file_name + file_ext, 'r') as f:
    text = f.read()


def preprocess_text(text):
    """Retain only alphanumeric characters in lowercase"""
    output = ""
    for line in text.split("\n"):
        if line:
            line = re.sub("-", " ", line).lower()
            line = re.sub("[^a-zA-Z0-9. \n]", "", line)
            output += line + " "

    return output

# Taking a subset of words
sub = 10 / 100  # percent
text = text[:int(sub * len(text))]

processed_text = preprocess_text(text)

with open(file_name + "_" + file_ext, "w") as f:
    f.write(processed_text)

# Calculating vocabulary size
processed_text = processed_text.replace(".", "")
vocabulary = list(set(processed_text.split()))
print("Vocabulary size:", len(vocabulary))
# print(vocabulary)
