import numpy as np
import pyANOVAapprox as ANOVAapprox
from TestFunctionPeriodic import *
import csv

rng = np.random.default_rng()

d = 10
def TestFunction(x):
    return b_spline_2(x[0])*b_spline_4(x[1])*b_spline_6(x[2]) + b_spline_2(x[3])*b_spline_4(x[4]) + b_spline_6(x[4])*b_spline_2(x[5]) + b_spline_4(x[5])*b_spline_6(x[6]) + b_spline_2(x[6])*b_spline_4(x[7]) + b_spline_6(x[7])*b_spline_2(x[8]) + b_spline_4(x[8])*b_spline_6(x[9])

M = 100_000
M_test = 1_000_000

Us = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (0,1), (0,2), (1,2),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(0,1,2)]

lmbda = 0.0

X = rng.random((M, d))
y = np.array([TestFunction(X[i, :].T) for i in range(M)], dtype=complex)  
X = X - 0.5
X_test = rng.random((M_test, d))
y_test = np.array([TestFunction(X_test[i, :].T) for i in range(M_test)], dtype=complex) 
X_test = X_test - 0.5

it = 9

ads = ANOVAapprox.approx(X, y, U=Us, basis="per")
ads.autoapproximate(maxiter=it, verbose=False)

for i in range(it):
    print("L2 error in iteration",str(i),":",str(ads.get_mse(lam=0.0,settingnr=i,X=X_test,y=y_test)))
    print("Bandwidths in iteration",str(i),":",str({ads.getSetting(i).U[j]:ads.getSetting(i).N[j] for j in range(len(ads.getSetting(1).U))}))
    
with open('plots/plotdata/s5_rates_l2error.csv', 'w') as csvfile:
    csvwrite = csv.writer(csvfile, delimiter=',')
    for i in range(it):
        csvwrite.writerow([i,ads.get_mse(lam=0.0,settingnr=i,X=X_test,y=y_test)])
        
for i in range(it):
    with open('plots/plotdata/s5_rates_bw_it'+str(i)+'.csv', 'w') as csvfile:
        csvwrite = csv.writer(csvfile, delimiter=',')
        csvwrite.writerow([{ads.getSetting(i).U[j]:ads.getSetting(i).N[j] for j in range(len(ads.getSetting(1).U))}])
