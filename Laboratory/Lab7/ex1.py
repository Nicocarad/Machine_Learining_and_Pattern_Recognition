import numpy as np
import scipy.optimize as opt

#the function accepts an array of variables 
def f(x):
    y = x[0]
    z = x[1]
    
    funct = (y+3)**2 + np.sin(y) + (z + 1)**2 #two dimensional function
    

    
    return funct


if __name__ == '__main__':
    
    
   start = np.array([0,0]) 
   # fmin_l_bfgs_b evaluates the minimum of the function with a numerical solution
   # start is array of coordinates where the algorithm starts searching the minimum 
   x,fun,d = opt.fmin_l_bfgs_b(f,start,approx_grad = True)
   print(x)

