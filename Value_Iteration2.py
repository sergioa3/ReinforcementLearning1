import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt

epsilon = 0.01

def value_iteration(S, A, P, R):
    k = epsilon*(1-lam)/(2*lam)
    V = {s: 0 for s in S}
    optimal_policy = {s: 0 for s in S}
    numero_it = []
    VectoresValor = []
    n = 0
    while True:
        oldV = V.copy()
        numero_it.append(n)
        VectoresValor.append(oldV)
        print(n)
        n = n + 1
        for s in S:
            Q = {}
            for a in A:
                Q[a]= R(s,a) + lam*sum(P(s_next,s,a) * oldV[s_next] for s_next in S)
                
            #print(s)
            V[s] = max(Q.values())
            optimal_policy[s] = max(Q, key=Q.get)

        
        if all(oldV[s] <= V[s] + k and oldV[s] >= V[s] - k  for s in S):
            break
        
    return numero_it, VectoresValor, optimal_policy

#-- constantes
lam=0.9
r=10
c=2
p=3
#-- limite de carros en cada sucursal
N1 = p
N2 = p
#-- limite de clientes en cada sucursal
Cl = p

#-- estados 
S = []

for i in range(0,N1+1):
    for j in range(0,N2+1):
        S.append((i,j))

A = [i for i in range(-N1,N2+1)]



def f1(x):
    return poisson.pmf(x,3)
def f2(x):
    return poisson.pmf(x,4)
def g1(x):
    return poisson.pmf(x,3)
def g2(x):
    return poisson.pmf(x,2)

def P(s_next, s, t):
    s1,s2 = s
    s1_next,s2_next = s_next
    res = 0
    if t<=s1 and -s2<=t:
        res = sum(f1(c1)*g1(s1_next-(s1-t-min(s1-t,c1))) for c1 in range(0,Cl+1)) *sum(f2(c2)*g2(s2_next-(s2+t-min(s2+t,c2))) for c2 in range(0,Cl+1))
    return res

def R(s,t):
    s1,s2 = s
    res = 0
    if t<=s1 and -s2<=t:
        res = (sum(f1(c)*min(s1-t,c) for c in range(0,Cl+1))
        +
        sum(f2(c)*min(s2+t,c) for c in range(0,Cl+1))
        )*r - abs(t)*c 
    
    return res

    


X,Z, optPol = value_iteration(S, A, P, R)
for x in X:
    print(x)


#print(v)

Y = []
#Z es una lista de diccionarios
for v in Z:
    print(max(v.values()))
    dif_v = [abs(v[s]-Z[-1][s]) for s in S]
    norma_dif_v = max(dif_v)
    Y.append(norma_dif_v)
    





for cosa in optPol:
    print('Estado:',cosa,', Accion',optPol[cosa])
'''optPol[(0,0)]
matriz = np.matrix([10,10])
for key in optPol:
    matriz[key[0]][key[1]] = optPol[key]

print(matriz)'''

plt.plot(X, Y)
plt.xlabel("n")
plt.ylabel("||Vn-V*||")
plt.show()

    

