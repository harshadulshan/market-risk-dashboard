
def calculate_total_sales(sales):
    """Return the total of all sales amounts."""
    return sum(sales)


def calculate_average_sales(sales):
    """Return the average sale, or zero if there are no sales."""
    if not sales:
        return 0

    return sum(sales) / len(sales)
