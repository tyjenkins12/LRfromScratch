import numpy as np
from numpy import genfromtxt, array

def compute_error(b, m, points):

    # Computing the mean squared error from the data points to the predicted regression line
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2

    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)

    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    
    # Initializing the gradient at 0
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))

    # Calculating the gradient based on points
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / n) * (y -((m_current * x) + b_current))
        m_gradient += -(2 / n) * x * (y - ((m_current * x) + b_current))

    # Assigning the new gradient for the next loop
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]

def run():

    # Collecting the data
    points = genfromtxt('data.csv', delimiter = ',')

    # Defining the rate of convergence for the model
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, points)))

if __name__ == '__main__':
    run()