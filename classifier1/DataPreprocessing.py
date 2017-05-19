from collections import Counter


def remove_categories_with_less_than_n(x_y_tuples, n):
    filtered = {}
    for category, (X, Y) in x_y_tuples.items():
        frequency_count = Counter(Y)
        if frequency_count[1] >= n:
            filtered[category] = (X.copy(),Y.copy())
    return filtered