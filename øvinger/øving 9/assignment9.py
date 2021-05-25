# importing the right packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


# main function to simulate the system
def simulate( a, b, K, lambda2, sigma2, y0, N, reference_frequency = 0 ):

    # storage allocation
    y = np.zeros(N)
    u = np.zeros(N)

    # saving the initial condition
    y[0] = y0
    
    # system noises
    e = np.random.normal(0, np.sqrt(lambda2),  N)
    r = np.random.normal(0, np.sqrt(sigma2), N) +     \
        np.sin( reference_frequency * np.arange(N) )
    
    # cycle on the steps
    for t in range(1, N):
        y[t] = b*u[t - 1] + e[t] + - a*y[t - 1]
        u[t] = -K*y[t] + r[t]

    return [y, u]


# define also a function for doing poles allocation, considering
# that eventually if the reference is absent then the ODE is
#
# y_k + ( a + b K ) y_{k-1} = e_k
#
def compute_gain( a, b, desired_pole_location ):
    
    return (desired_pole_location + a)/(-b)
    #return desired_pole_location


# plotting of the impulse response
def plot_impulse_response( a, b, figure_number = 1000 ):
    
    # ancillary quantities
    k = range(0,50)
    y = b * np.power( -a, k )
    
    # plotting the various things
    plt.figure( figure_number )
    plt.plot(y, 'r-', label = 'u')
    plt.xlabel('time')
    plt.ylabel('impulse response relative to a = {} and b = {}'.format(a, b))

# define the system parameters
a = -0.5
b = 2
K = compute_gain( a, b, 0.7 )

# noises
lambda2 = 0.1 # on e
sigma2  = 0.1 # on r

# initial condition
y0 = 3

# number of steps
N = 100


# DEBUG - check that things work as expected

# run the system
[y, u] = simulate( a, b, K, lambda2, sigma2, y0, N, 0.3 )

# plotting the various things
plt.figure()
plt.plot(y[:-1], 'k:', label = 'y')
plt.plot(u[:-1], 'r-', label = 'u')
plt.xlabel('time')
plt.ylabel('signals')
plt.legend()


plt.figure()
plot_impulse_response( a, b )
plt.show()


