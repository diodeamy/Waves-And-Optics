import numpy as np
from scipy.odr import ODR, Model, Data, RealData
import matplotlib.pyplot as plt

def fit(data, model="linear", n=1, supress_plot=False, supress_output=False,
        title="Plot with fit", xlabel="x-axis", ylabel="y-axis"):
    '''
    Fits a model to datapoints and corresponding errors, saving a plot of the data and fit.
    Returns best fit parameters popt and corresponding covariance matrix pcov
    INPUTS
    -args
        data:   n-by-4 array xdata, ydata, xerror, yerror
    -kwargs
        model:  string, model to fit on data, possible values are
            {"linear", "polynomial", "exponential"}
        n:      if model=="polynomial", degree of model polynomial
        supress_plot:   bool, if True does not plot data and fit
        supress_output: bool, if True does not return poopt and pcov
    OUTPUTS
        param:  optimal fit parameters
        error:  error on optimal fit parameters
    '''
    #set up data to fit using odr
    odrData = RealData(data[:,0],data[:,1],data[:,2],data[:,3])

    #chose model to fit on data
    if model=="linear":
        #set up linear model through the origin
        func = lambda param,x: param*x
        model = Model(func)
        
    elif model=="polynomial":
        #set up degree n polynomial model
        power = np.arange(0,n+1)
        func = lambda param,x: (param*x**power).sum
        model = Model(func)

    elif model=="exponential":
        #set up exponential model
        func = lambda param,x: param[0]*np.exp(param[1]*x)
        model = Model(func)

    #set up ODR method
    odr = ODR(odrData, model)
    odr.set_job(fit_type=0)

    #compute fit parameters and their errors
    output = odr.run()
    param, error = output.beta, output.sd_beta

    fit = lambda x: model(param,x)

    #plot data, errorbars on data and fit on data
    if not supress_plot:
        plt.figure(figsize=(9,7))
        plt.grid(True)
        
        plt.xlabel("xlabel")
        plt.ylabel("ylabel")
        plt.title("title")

        plt.errorbar(data[:,0],data[:,1],data[:,3],data[:,2],fmt="or",label="data with errorbars")
        x = np.linspace(data[0,0],data[0,-1],1000)
        plt.plot(x,fit(x),fmt="-b",label="linear fit on data")

        plt.legend()

        plt.show()

    if not supress_output:
        return param, error