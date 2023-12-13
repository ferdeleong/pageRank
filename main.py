import numpy as np

#Dictionay
links = {
    1: [2, 3, 7, 8],
    2: [1, 4, 5, 9],
    3: [2, 4, 8],
    4: [1, 3, 5],
    5: [2, 4, 7, 8],
    6: [2, 3, 5, 10],
    7: [1, 4, 6, 9],
    8: [2,3,7,8],
    9: [1, 3, 5, 7],
    10: [2, 4, 6, 9],
}

# Find the maximum page number from the links
max_page = max(links.keys())

# Create a matrix H filled with zeros of size (max_page x max_page)
H = np.zeros((max_page, max_page))

# Fill the matrix H based on links information
for i, outbound_links in links.items():
    for j in outbound_links:
        H[i - 1][j - 1] = round(1 / len(outbound_links), 3)


print(H)

# Number of pages
num_pages = len(links)

# Create a matrix r filled with 1/num_pages
r = [[1 / num_pages] * num_pages for _ in range(num_pages)]

# Display the matrix r
for row in r:
    print(row)