import sys
import math


def get_parameter_vectors():
    """
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    described in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    """
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0]*26
    s = [0]*26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char)-ord('A')] = float(prob)
    f.close()

    return e, s


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    Dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0,
            'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}
    X = Dict
    with open(filename, encoding='utf-8') as f:
        # TODO: add your code here
        for line in f:
            string = line.strip()
            for element in range(0, len(string)):
                char = string[element].upper()
                if char >= 'A' and char <= 'Z':
                    X[char] = X[char] + 1
    f.close()

    return X


# TODO: add your code here for the assignment

# (Q1)
# Print “Q1” followed by the 26 character counts for letter.txt.
# Please see sample output files for the exact format.
# We will test it on different letter.txt to check your count output.
print("Q1")
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
letter_count = shred("letter.txt")
for i in alphabet:
    output_cast = i + " " + str(letter_count.get(i))
    print(output_cast)

# (Q2)
# Compute X1*log(e1) and X1*log(s1).
# Print “Q2” then these values up to 4 decimal places on two separate lines
print("Q2")
tuples = get_parameter_vectors()
english_prob_a = letter_count.get('A') * math.log(tuples[0][0])
spanish_prob_a = letter_count.get('A') * math.log(tuples[1][0])
print("{:.4f}".format(english_prob_a))
print("{:.4f}".format(spanish_prob_a))

# (Q3)
# Compute F(English) and F(Spanish).
# Print “Q3” followed by their values up to 4 decimal places on two separate lines.
print("Q3")
prior_english = 0.6
log_prior_english = math.log(prior_english)
sigma_english = 0
j = 0
for i in alphabet:
    iteration = letter_count.get(i) * math.log(tuples[0][j])
    sigma_english += iteration
    j += 1
F_english = log_prior_english + sigma_english
print("{:.4f}".format(F_english))

prior_spanish = 0.4
log_prior_spanish = math.log(prior_spanish)
sigma_spanish = 0
k = 0
for i in alphabet:
    iteration = letter_count.get(i) * math.log(tuples[1][k])
    sigma_spanish += iteration
    k += 1
F_spanish = log_prior_spanish + sigma_spanish
print("{:.4f}".format(F_spanish))

# (Q4)
# Compute P(Y = English | X).
# Print “Q4” then this value up to 4 decimal places.
print("Q4")
P_english = 0
if (F_spanish - F_english) >= 100:
    P_english = 0
elif (F_spanish - F_english) <= -100:
    P_english = 1
else:
    P_english = (1 / (1 + (math.exp(F_spanish - F_english))))
print("{:.4f}".format(P_english))
