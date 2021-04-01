# matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace, bernoulli
import scipy.optimize as optimize
import math
from prettytable import PrettyTable

N = 10
magnitude = 1.2
mu = 0  # loc
sigma = 1  # beta
alpha = 0
gamma = 0.1
x = np.linspace(-1, 1, N)
params = [1, -8, 4, 3, 5]
model_order = np.linspace(0,9,10)

def create_y_model(params):
    def y(x): return sum([p*(x**i) for i, p in enumerate(params)])
    return y


def add_noise_to_true_model(x,N, params, magnitude, alpha, mu, sigma):
    y = create_y_model(params)
    y_true = y(x)
    y_added_noise = y_true + magnitude * \
        (alpha*np.random.normal(mu, sigma, N) +
         (1-alpha)*np.random.laplace(mu, sigma, N))
    return y_added_noise



y_measured_data = add_noise_to_true_model(x,N, params, magnitude, alpha, mu, sigma)
plt.figure()
plt.plot(x, y_measured_data, label="Measured data")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
#plt.show()


#define traning, testing and validation set
y_traning_set = y_measured_data[0:math.floor(len(y_measured_data)*1/2)]
x_traning_set = x[0:math.floor(len(x)/2)]
y_testing_set = y_measured_data[math.floor(len(y_measured_data)*1/2):]
x_testing_set = x[math.floor(len(x)*1/2):]


#Estimating the LS estimator for each mode
def LS_estimator_given_mode(mode, x, y):
    u_tensor_0 = np.reshape(x, (len(x), 1))

    
    ones_vec = np.ones((len(x), 1))
    u_tensor = np.append(ones_vec, u_tensor_0, axis=1)

    if mode ==1:
        u_tensor = ones_vec
    for i in range(2, mode):
        u_tensor = np.append(u_tensor, np.power(u_tensor_0, i), axis=1)

    u_transpose_dot_u = np.dot(u_tensor.T, u_tensor)  # calculating dot product
    u_transpose_dot_u_inv = np.linalg.inv(
        u_transpose_dot_u)  # calculating inverse
    u_transpose_dot_y = np.dot(u_tensor.T, y)  # calculating dot product

    LS_params = np.dot(u_transpose_dot_u_inv, u_transpose_dot_y)
    # Recreate model based on LS estimate:
    LS_params = LS_params.tolist()
    return LS_params
# Estimating the ML estimator for each mode
def log_lik(par_vec,x,y):
    pdf = laplace.pdf
    # If the standard deviation parameter is negative, return a large value:
    if par_vec[-1] < 0:
        return(1e8)
    # The likelihood function values:
    lik = pdf(y,
              loc=sum([p*(x**i) for i, p in enumerate(par_vec[:-1])]),
              scale=par_vec[-1])

    if all(v == 0 for v in lik):
        return(1e8)
    # Logarithm of zero = -Inf
    return(-sum(np.log(lik[np.nonzero(lik)])))

def ML_estimator_given_mode(mode,x,y):
    init_guess = np.zeros(mode+1)
    init_guess[-1] = len(x)

    opt_res = optimize.minimize(fun=log_lik,
                                x0=init_guess,
                                options={'disp': False},
                                args=(x,y))
    MLE_params = opt_res.x[:-1]
    MLE_params = MLE_params.tolist()
    return MLE_params


#calculate 10 models for both LS and ML
def creating_different_model(x_data, y_data,data,type):
    if type == "ML":
        for i in range(len(data)):
            data[i] = ML_estimator_given_mode(i + 1,x_data,y_data)
            data[i]= create_y_model(data[i])
    if type == "LS":
        for i in range(len(data)):
            data[i] = LS_estimator_given_mode(i + 1,x_data,y_data)
            data[i]= create_y_model(data[i])

    return data



y_hat_ML_10_models = [0] *10
y_hat_LS_10_models = [0] *10
#Function declates the all the models and then estiamtes the model paramters based on given data
creating_different_model(x, y_measured_data,y_hat_LS_10_models,"LS")
creating_different_model(x_traning_set, y_traning_set,y_hat_ML_10_models,"ML")



#calculate performacne index
def rmse_performance_index(x_data,y_data,y_hat_models):
    performance_vector = [0]*len(y_hat_models)
    for i in range(len(y_hat_models)):
        y_hat = y_hat_models[i](x_data)
        for j in range(len(y_data)):
            performance_vector[i] += (abs(y_data[j] - y_hat[j]))**2
        performance_vector[i] = (performance_vector[i]/len(y_data))**(1/2)
        #print(performance_vector[i])
    return performance_vector


def rss_performance_index(x_data,y_data,y_hat_models):
    performance_vector = [0]*len(y_hat_models)
    for i in range(len(y_hat_models)):
        y_hat = y_hat_models[i](x_data)
        for j in range(len(y_data)):
            performance_vector[i] += (abs(y_data[j] - y_hat[j]))**2
        #print(performance_vector[i])
    return performance_vector


def mean(x_data,y_data,y_hat_models):
    mean = [0]*len(y_hat_models)
    for i in range(len(y_hat_models)):
        y_hat = y_hat_models[i](x_data)
        for j in range(len(y_hat)):
            mean[i] += y_hat[j]
        mean[i] = mean[i]/len(y_hat)
    return mean


def fvu_performance_index(x_data,y_data,y_hat_models):
    mean_vector = mean(x_data,y_data,y_hat_models)
    rss_performance_vector = rss_performance_index(x_data,y_data,y_hat_models)
    fvu_performance_vector = [0]*len(y_hat_models)

    variance_vector = [0]*len(y_hat_models)
    for i in range(len(y_hat_models)):
        for j in range(len(y_data)):
            variance_vector[i] += (abs(y_data[j] - mean_vector[i]))**2
        fvu_performance_vector[i] = rss_performance_vector[i]/variance_vector[i]
    return fvu_performance_vector


def rr_performance_index(x_data,y_data,y_hat_models):
    fvu_performance_vector = fvu_performance_index(x_data,y_data,y_hat_models)
    rr_performance_vector = [0]*len(y_hat_models)

    for i in range(len(y_hat_models)):
        rr_performance_vector[i] = 1 - fvu_performance_vector[i]

    return rr_performance_vector


def fit_performance_index(x_data,y_data,y_hat_models):
    fvu_performance_vector = fvu_performance_index(x_data,y_data,y_hat_models)
    fit_performance_vector = [0]*len(y_hat_models)

    for i in range(len(y_hat_models)):
        fit_performance_vector[i] = 100*(1 - (fvu_performance_vector[i])**(1/2))
    return fit_performance_vector


def make_table(x_data,y_data):
    ML_rmse_performance_vector = rmse_performance_index(x_data,y_data,y_hat_ML_10_models)
    ML_rss_performance_vector = rss_performance_index(x_data,y_data,y_hat_ML_10_models)
    ML_fvu_performance_vector = fvu_performance_index(x_data,y_data,y_hat_ML_10_models)
    ML_rr_performance_vector  = rr_performance_index(x_data,y_data,y_hat_ML_10_models)
    ML_fit_performance_vector = fit_performance_index(x_data,y_data,y_hat_ML_10_models)

    LS_rmse_performance_vector = rmse_performance_index(x_data,y_data,y_hat_LS_10_models)
    LS_rss_performance_vector = rss_performance_index(x_data,y_data,y_hat_LS_10_models)
    LS_fvu_performance_vector = fvu_performance_index(x_data,y_data,y_hat_LS_10_models)
    LS_rr_performance_vector  = rr_performance_index(x_data,y_data,y_hat_LS_10_models)
    LS_fit_performance_vector = fit_performance_index(x_data,y_data,y_hat_LS_10_models)

    x = PrettyTable()
    x.add_column("Paramter order(MLE)",model_order)
    x.add_column("RMSE",ML_rmse_performance_vector)
    x.add_column("RSS",ML_rss_performance_vector)
    x.add_column("FVU",ML_fvu_performance_vector)
    x.add_column("RR",ML_rr_performance_vector)
    x.add_column("FIT",ML_fit_performance_vector)
    print(x)

    t = PrettyTable()
    t.add_column("Paramter order(LS)",model_order)
    t.add_column("RMSE",LS_rmse_performance_vector)
    t.add_column("RSS",LS_rss_performance_vector)
    t.add_column("FVU",LS_fvu_performance_vector)
    t.add_column("RR",LS_rr_performance_vector)
    t.add_column("FIT",LS_fit_performance_vector)
    print(t)

#Calculate the various performance indexes on the training and the testing set and make a table showing all the values.
#make_table(x_traning_set,y_traning_set)
#make_table(x_testing_set,y_testing_set)


def plotting(x_data,y_data):
    ML_rmse_performance_vector = rmse_performance_index(x_data,y_data,y_hat_ML_10_models)
    ML_rss_performance_vector = rss_performance_index(x_data,y_data,y_hat_ML_10_models)
    ML_fvu_performance_vector = fvu_performance_index(x_data,y_data,y_hat_ML_10_models)
    ML_rr_performance_vector  = rr_performance_index(x_data,y_data,y_hat_ML_10_models)
    ML_fit_performance_vector = fit_performance_index(x_data,y_data,y_hat_ML_10_models)

    LS_rmse_performance_vector = rmse_performance_index(x_data,y_data,y_hat_LS_10_models)
    LS_rss_performance_vector = rss_performance_index(x_data,y_data,y_hat_LS_10_models)
    LS_fvu_performance_vector = fvu_performance_index(x_data,y_data,y_hat_LS_10_models)
    LS_rr_performance_vector  = rr_performance_index(x_data,y_data,y_hat_LS_10_models)
    LS_fit_performance_vector = fit_performance_index(x_data,y_data,y_hat_LS_10_models)

    plt.figure()
    plt.title("ML estimators")
    plt.xlabel("Model orders")
    plt.plot(model_order, ML_rmse_performance_vector, label= "RMSE performance")
    plt.plot(model_order, ML_rss_performance_vector, label= "RSS performance")
    plt.plot(model_order, ML_fvu_performance_vector, label= "FVU performance")
    plt.plot(model_order, ML_rr_performance_vector, label= "RR performance")
    plt.plot(model_order, ML_fit_performance_vector, label= "FIT performance")
    plt.ylim(0, 300)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.title("LS estimators")
    plt.xlabel("Model orders")
    plt.plot(model_order, LS_rmse_performance_vector, label= "RMSE performance")
    plt.plot(model_order, LS_rss_performance_vector, label= "RSS performance")
    plt.plot(model_order, LS_fvu_performance_vector, label= "FVU performance")
    plt.plot(model_order, LS_rr_performance_vector, label= "RR performance")
    plt.plot(model_order, LS_fit_performance_vector, label= "FIT performance")
    plt.ylim(0, 50)
    plt.legend()
    plt.tight_layout()
    plt.show()


plotting(x_traning_set,y_traning_set)

plotting(x_testing_set,y_testing_set)