import numpy as np
import scipy.optimize as opt

def f(x):
    y = x[0]
    z = x[1]
    
    funct = (y+3)**2 + np.sin(y) + (z + 1)**2
    

    
    return funct


if __name__ == '__main__':
    
    
   start = np.array([0,0]) 
   
   x,fun,d = opt.fmin_l_bfgs_b(f,start,approx_grad = True)
   print(x)

