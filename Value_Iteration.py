import numpy as np
from scipy.stats import poisson


def value_iteration(S, A, P, R):
    V = {s: 0 for s in S}

    while True:
        oldV = V.copy()

        for s in S:
            print(s)
            Q = {}
            for a in A:
                Q[a]= R(s,a) + lam*sum(P(s_next,s,a) * oldV[s_next] for s_next in S)
            
            V[s] = max(Q.values())
        
        if all(oldV[s] == V[s] for s in S):
            break
        
    return V

#-- constantes
lam=0.9
r=10
c=2

#-- limite de carros en sucursales
N = 2

#-- limite de clientes en sucursales
Cl = 4

#-- estados 
S = []

for i in range(0,N):
    for j in range(0,N):
        for m in range(0,Cl):
            for k in range(0,Cl):
                S.append((i,j,m,k))

A = [i for i in range(0,N)]


def f1(x):
    return poisson.pmf(x,3)
def f2(x):
    return poisson.pmf(x,4)
def g1(x):
    return poisson.pmf(x,3)
def g2(x):
    return poisson.pmf(x,2)

def P(s_next, s, t):
    s1,s2,c1,c2 = s
    s1_next,s2_next,c1_next,c2_next = s_next
    if t<=s1 and -s2<=t:

        return f1(c1_next)*f2(c2_next)*g1(s1_next-(s1-t-min(s1-t,c1)))*g2(s2_next-(s2+t-min(s2+t,c2)))
    else:
        return 0

def R(s,t):
    s1,s2,c1,c2 = s
    
    if t<=s1 and -s2<=t:
        return (min(s1-t,c1)+min(s2+t,c2))*r - abs(t)*c
    else:
        return -9999999




v = value_iteration(S, A, P, R)

print(v)


