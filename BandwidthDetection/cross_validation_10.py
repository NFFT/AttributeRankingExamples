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

sigma = 0.01

X = rng.random((M, d))
f = np.array([TestFunction(X[i, :].T) for i in range(M)], dtype=complex)
X = X - 0.5
y = f + sigma * np.random.standard_normal(M) * (np.max(f) - np.min(f))

print("sigma =",str(np.sqrt(np.linalg.norm(f-y)**2/len(f))))

with open('plots/plotdata/s10_cv_sigma.csv', 'w') as csvfile:
    csvwrite = csv.writer(csvfile, delimiter=',')
    csvwrite.writerow([np.sqrt(np.linalg.norm(f-y)**2/len(f))]) 

X_test = rng.random((M_test, d))
y_test = np.array([TestFunction(X_test[i, :].T) for i in range(M_test)], dtype=complex) 
X_test = X_test - 0.5

it = 3

ads = ANOVAapprox.approx(X, y, U=Us, basis="per")
setting = ads.getSetting()

Bmin = sum(5**len(u) for u in Us)
Bmax = M/3
Bs = np.geomspace(Bmin, Bmax, num=10)

D = dict([(u, tuple([1.0] * len(u))) for u in Us])
t = dict([(u, tuple([1.0] * len(u))) for u in Us])

for idx in range(it):
    print("Iteration",idx)
    print("B cv mse")
    cv = np.zeros(len(Bs))
    L2error = np.zeros(len(Bs))
    
    for idx_cv in range(len(Bs)):
        bw = ANOVAapprox.compute_bandwidth(Bs[idx_cv], D, t)
        ads.addSetting(setting)
        ads.getSetting().N = [bw[i] for i in Us]
        ads.addTrafo()
        ads.approximate()
        
        nfreqs = np.sum([np.prod(np.array(bw[u])-1) for u in Us])+1
        cv[idx_cv] = 1/M*np.linalg.norm(ads.evaluate(X=X,lam=0.0)-y)**2/(1-nfreqs/M)**2
        
        L2error[idx_cv] = np.sqrt(ads.get_mse(X=X_test,y=y_test,lam=0.0))
        
        print(Bs[idx_cv], cv[idx_cv], L2error[idx_cv])
    
    with open('plots/plotdata/s10_cv_it'+str(idx+1)+'.csv', 'w') as csvfile:
        csvwrite = csv.writer(csvfile, delimiter=',')
        for idx_cv in range(len(Bs)):
            csvwrite.writerow([Bs[idx_cv], cv[idx_cv], L2error[idx_cv]]) 
        
    B = Bs[np.argmin(cv)]
    bw = ANOVAapprox.compute_bandwidth(B, D, t)
    ads.addSetting(setting)
    ads.getSetting().N = [bw[i] for i in Us]
    ads.addTrafo()
    ads.approximate()
    D, t = ads.estimate_rates(lam=0.0)
