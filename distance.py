# calculate the distance on the following cities as a ring:
#points = [[10, 2], [8, 4], [6, 3], [5, 2], [4, 3], [2, 4], [1, 2], [0, 0], [3, 1], [7, 1], [9, 0]]
points = [[7, 1], [9, 0], [10, 2], [8, 4], [6, 3], [5, 2], [4, 3], [2, 4], [1, 2], [0, 0], [3, 1]]

distance = 0
for i in range(len(points)-1):
    distance += ((points[i][0] - points[i + 1][0])**2 + (points[i][1] - points[i + 1][1])**2)**0.5

distance += ((points[-1][0] - points[0][0])**2 + (points[-1][1] - points[0][1])**2)**0.5
print(distance)