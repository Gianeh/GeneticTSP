# bruteforce on traveling salesman problem
import itertools
import time

# 11 cities
cities_11 = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0), (4.0, 3.0), (2.0, 4.0), (5.0, 2.0), (6.0, 3.0), (7.0, 1.0), (8.0, 4.0), (9.0, 0.0), (10.0, 2.0)]

# 13 cities
cities_13 = [(0.5, 3.0), (6.0, 2.0), (7.3, 2.0), (12.0, 4.0), (3.0, 5.0), (5.0, 10.0), (5.0, 12.0), (5.0, 0.0), (9.0, 4.0), (15.0, 13.0), (10.0, 2.0), (1.0, 3.0), (12.0, 1.0)]

def distance(city1, city2):
    return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5

def total_distance(cities, order):
    result = 0
    for i in range(len(order) - 1):
        result += distance(cities[order[i]], cities[order[i + 1]])
    result += distance(cities[order[-1]], cities[order[0]])
    return result

def solve(cities):
    best_order = []
    best_distance = 0
    for order in itertools.permutations(range(len(cities))):
        d = total_distance(cities, order)
        if best_order == [] or d < best_distance:
            best_order = order
            best_distance = d
    return best_order, best_distance

start = time.time()

best_order, best_distance = solve(cities_11)

end = time.time()

#tour = [cities[i] for i in best_order]
tour = [cities_13[i] for i in best_order]
print(tour, best_distance)

print("Time: ", (end - start)/60, "minutes")

