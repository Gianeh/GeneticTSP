import matplotlib
matplotlib.use('TkAgg')  # Specify the backend
import matplotlib.pyplot as plt
import os

# Function to read points from file
def read_points_from_file(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = line.split()
            points.append((float(x), float(y)))
    points.append(points[0])  # Connect the last point with the first point
    return points

# Function to plot points and connect them with lines
def plot_points_with_lines(points):
    x_values, y_values = zip(*points)
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points Connected by Lines')
    plt.grid(True)
    # keep the window open
    plt.show()


# Main function
def main():
    # best_solution.txt path
    path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(path, '../../results/best_solution.txt')
    print(filename)
    
    points = read_points_from_file(filename)
    print(points)
    plot_points_with_lines(points)

if __name__ == "__main__":
    main()
