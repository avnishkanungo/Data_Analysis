from numpy import *
from pandas import *

def compute_error_for_points(b, m , points):
    totalError = 0
    #for i in range(0 , len(points)):
    x = points[:,0:2]
    t = x.shape[0]
    y = points[:,2:0]
    totalError += (y-((m * t) + b) ** 2)
    return totalError.sum()/float(len(points))

def gradient_descent_step(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    #for i in range(0, len(points)):
    x = points[:,0:2]
    t_current = x.shape[0]
    y = points[:,2:0]
    b_gradient += -(2/N)*(y-(((m_current * t_current)) + b_current))
    m_gradient += -(2/N)* t_current * (y-(((m_current * t_current)) + b_current))
    new_b = b_current - (learning_rate * b_gradient.sum())
    new_m = m_current - (learning_rate * m_gradient.sum())
    return [new_b,new_m]


def gardient_descent(b_start,m_start, points, learning_rate, num_iteration):
    b = b_start
    m = m_start
    for i in range(int(num_iteration)):
        b,m = gradient_descent_step(b,m,points,learning_rate)
    return [b,m]

def run():
    points = genfromtxt("/Users/avnish/Downloads/machine-learning-ex1/ex1/ex1data2.txt", delimiter = ",")
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iteration = 1000
    print "Gradient Descent starting b={0}, m={1} and error={2}".format(initial_b, initial_m, compute_error_for_points(initial_b,initial_m,points))
    print "Running..."
    [b,m] = gardient_descent(initial_b,initial_m,points,learning_rate,num_iteration)
    print " At end of Gradient Descent with {0} iterations b={1},m={2} and error={3}".format(num_iteration,b, m, compute_error_for_points(b,m,points))


if __name__ == '__main__':
    run()
