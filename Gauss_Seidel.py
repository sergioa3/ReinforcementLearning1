from cProfile import label
import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt

epsilon = 0.01

def gauss_Seidel(S, A, P, R, lam):
    k = (epsilon*(1-lam))/(2*lam)
    V = {s: 0 for s in S}
    optimal_policy = {s: 0 for s in S}
    numero_it = []
    VectoresValor = []
    n = 0
    while True:
    
        oldV = V.copy()
        numero_it.append(n)
        VectoresValor.append(oldV)
        #print(n)
        n = n + 1
        for j in range(len(S)):
        
            Q = {}
            for a in A:
            
                Q[a]= R(S[j],a) + lam*(sum(P(s_next,S[j],a) * V[s_next] for s_next in S[:j])
                +sum(P(s_next,S[j],a) * oldV[s_next] for s_next in S[j:]))
                
            #print(s)
            V[S[j]] = max(Q.values())
            optimal_policy[S[j]] = max(Q, key=Q.get)

    
        if all(oldV[s] <= V[s] + k and oldV[s] >= V[s] - k  for s in S):
            break
        
    return numero_it, VectoresValor, optimal_policy, n



#-- constantes
lam=0.9
r=10
c=2
p=5
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

    



'''
X2,Z2, optPol2 = value_iteration(S, A, P, R)
for x in X:
    print(x)'''


'''#print(v)

Y = []
#Z es una lista de diccionarios
for v in Z:
    print(max(v.values()))
    dif_v = [abs(v[s]-Z[-1][s]) for s in S]
    norma_dif_v = max(dif_v)
    Y.append(norma_dif_v)'''
    
'''Y2 = []
#Z es una lista de diccionarios
for v in Z2:
    print(max(v.values()))
    dif_v = [abs(v[s]-Z[-1][s]) for s in S]
    norma_dif_v = max(dif_v)
    Y2.append(norma_dif_v)'''




'''for cosa in optPol:
    print('Estado:',cosa,', Accion',optPol[cosa])'''
'''for cosa in optPol2:
    print('Estado:',cosa,', Accion',optPol2[cosa])'''
'''optPol[(0,0)]
matriz = np.matrix([10,10])
for key in optPol:
    matriz[key[0]][key[1]] = optPol[key]

print(matriz)'''

#plt.plot(X, Y, label="Gauss Seidel")

#plt.plot(X2, Y2, label="Value Iteration")



lamdas = [0.4,0.5, 0.6, 0.7,0.8, 0.9, 0.95]
#resultadosX = []
resultadosVect = []
resultadosNormasVect = []
resultadosNumIts = []
for lam in lamdas:
    print('\n')
    print('\n')
    print('lambda = {}'.format(lam))
    X,Z, optPol, numIts = gauss_Seidel(S, A, P, R, lam)
#    resultadosX.append(X)
    #resultadosVect.append({s:(1-lam)*Z[-1][s] for s in Z[-1]})
    for s in Z[-1]:
        print('({})V*_{}{}={}'.format(1-lam,lam,s,(1-lam)*Z[-1][s]))
    resultadosNormasVect.append((1-lam)*max(Z[-1].values()))
    resultadosNumIts.append(numIts)
    print('Numero  de Iteraciones = {}'.format(numIts))






plt.plot(lamdas, resultadosNormasVect, label='(1-lambda)||V*||')
plt.plot(lamdas, resultadosNumIts, label='Numero Itns')
plt.xlabel("lambda")
plt.ylabel("")
plt.legend()
plt.show()

    

