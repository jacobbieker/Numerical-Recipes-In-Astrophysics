import math
import matplotlib.pyplot as plt

nums = [i for i in range(100)]

def avg(array):
    avg = 0
    for element in array:
        avg += element

    avg /= len(array)
    return avg


def std(array):
    std = 0
    mean = avg(array)
    for element in array:
        std += abs(element - mean)**2

    std /= len(array)
    return math.sqrt(std)

even_nums = [i for i in range(0, 100, 2)]

odd_nums = []

for element in nums:
    if element not in even_nums:
        odd_nums.append(element)

avg(even_nums)
std(even_nums)

avg(odd_nums)
std(odd_nums)

excluded_nums = [i for i in range(10,20,1)]
other_excluded = [i for i in range(45,57,1)]

part_c_array = []

for element in nums:
    if element not in excluded_nums and element not in other_excluded:
        part_c_array.append(element)

avg(part_c_array)
std(part_c_array)

part_d_array = []
for element in nums:
    if element in excluded_nums or element in other_excluded:
        part_d_array.append(element)

avg(part_d_array)
std(part_d_array)

# Exercise 1 Part E

x = []

step = 3/100
end = 3
start = 0

for i in range(100):
    x.append(start+(step*i))

print(x)
A = 0.8
B = 2

y = []
for element in x:
    y.append(A*math.exp(element) - B*element)

plt.scatter(x, y)
plt.title("Part E")
#plt.show()

print("Mean of Y: {}".format(avg(y)))


# Part F

def factorial(number):
    if number == 1:
        return 1
    else:
        return number * factorial(number-1)

step = 3/10

x = []

for i in range(10):
    x.append(start+(step*i))

def estimate_ex(x, k):
    final_solution = 0
    final_solution += (x**k)/factorial(k)

y_est = []

for element in x:
    y_est.append(estimate_ex(element, 5))

print(len(x))
print(len(y))
plt.scatter(x, y_est, c='red')
plt.show()

# Seems to compare the same. Built-in should be faster.

math.factorial(5)

# Number 3

import numpy as np

def func_a(x):
    return 2*x*np.sin(0.8*x)

seven_points = np.linspace(1,20, 7)

all_points = [(x, func_a(x)) for x in seven_points]

def determine_closest_points(points, x_point):
    '''
    Determine the closest points for them
    :param points:
    :param x_point:
    :return:
    '''

    point_values = []
    y_values = []
    for index, point in enumerate(points):
        if index+1 > len(points):
            if points[index] < x_point:
                point_values.append(point)
        if points[index][0] < x < points[index+1][0]:
            point_values.append(points[index])
            point_values.append(points[index+1])

    return point_values

def linear_interp(points, new_x_points):
    '''
    Lienarly interpolate for the points
    :param points:
    :param new_x_points:
    :return:
    '''
    new_points = []
    for x_point in new_x_points:
        point_values = determine_closest_points(points, x_point)
        if len(point_values) == 1:
            # Now extrapolating, not interpolating
            return NotImplementedError
        else:
            new_y = ((point_values[1][1] - point_values[0][1])/(point_values[1][0] - point_values[0][0]))*(x_point - point_values[0][0])

        new_points.append((x_point, new_y))

    return new_points

def polynomial_interp(points, new_x_points):
    '''
    Polynomially interpolate points
    :param points:
    :param new_x_points:
    :return:
    '''

    new_points = []
    for x_point in new_x_points:
        point_values = determine_closest_points(points, x_point)
        if len(point_values) == :
            return NotImplementedError
        else:
            new_y = 0

        new_points.append((x_point, new_y))

def neville_interp(points, new_x_points):
    return NotImplementedError

