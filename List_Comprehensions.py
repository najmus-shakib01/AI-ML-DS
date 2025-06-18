numbers = [1, 2, 3, 4, 5]
words = ['Hello', 'World', 'Python']
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# squared_numbers = [x**2 for x in numbers]
# print(squared_numbers)

# even_numbers = [x for x in numbers if x % 2 != 0]
# print(even_numbers)

# upper_words = [word.lower() for word in words]
# print(upper_words)

# word_lengths = {word: len(word) for word in words}
# print(word_lengths)

# flattened = [num for row in matrix for num in row]
# print(flattened)

pairs = [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
print(pairs)