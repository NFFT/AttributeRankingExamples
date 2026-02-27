import numpy as np
import pyANOVAapprox as ANOVAapprox
import csv

from itertools import combinations

def get_superposition_set(d, ds):
    return [()] + [c for r in range(1, ds+1) for c in combinations(range(d), r)]

rng = np.random.default_rng()

d = 2

def TestFunction(x):
    def p2(x):
        return x**2-x+1/6
    def p4(x):
        return x**4-2*x**3+x**2-1/30
    return np.sqrt(378000/2281)*(p2(x[0])+p4(x[1])+p4(x[0])*p2(x[1]))
    
M = 100_000
M_test = 1_000_000

ds = 2
Us = get_superposition_set(d,ds)

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
    
with open('plots/plotdata/s3_rates_l2error.csv', 'w') as csvfile:
    csvwrite = csv.writer(csvfile, delimiter=',')
    for i in range(it):
        csvwrite.writerow([i,ads.get_mse(lam=0.0,settingnr=i,X=X_test,y=y_test)])
        
for i in range(it):
    with open('plots/plotdata/s3_rates_bw_it'+str(i)+'.csv', 'w') as csvfile:
        csvwrite = csv.writer(csvfile, delimiter=',')
        csvwrite.writerow([{ads.getSetting(i).U[j]:ads.getSetting(i).N[j] for j in range(len(ads.getSetting(1).U))}])
