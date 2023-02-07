import tensorflow as tf
import openpyxl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from integrationGTGBfunc import integrationfian
import matplotlib as plt



Npop = 100 #populasi
Maxit = 200 #iterasi
el = 0.05 #elitism
Pc = 0.8 #probabilitas crossover
Pm = 0.1 #probabilitas mutasi
Nvar = 8 #jumlah variabel desain yang dioptimasi
Nbit = 20 #jumlah bit

rb = [5.95, 5.83, 0, 0, 0, 8.3, 9.7, 0]
ra = [8.11, 8.38, 0.406944444, 3063, 51.5, 89, 14.5, 4.224140413]

eBangkit = []
Individu = []
eIndividu = []
david = []
Dadatfit = []
Datfit = []
summary = []
eDadatfit = []
efitnessmax = []
eIndividuMax = []

Bangkit = np.round(np.random.rand(Npop, Nbit * Nvar))
popsize = Bangkit.shape[0]
batas = np.zeros((Nvar))
for i in range(Nvar):
    batas[i] = ra[i] - rb[i]

Desimal = np.zeros((Npop, Nvar))
Individu = np.zeros((Npop, Nvar))
for i in range(Npop):
    for j in range(Nvar):
        slice = Bangkit[i][((j*Nbit)-(Nbit-1)):(j*Nbit)]
        binary_representation = ''.join(map(lambda x: '1' if x >= 0.5 else '0', slice))
        if binary_representation:
            Desimal[i][j] = int(binary_representation, 2)
        else:
            Desimal[i][j] = 0
        Individu[i][j] = (Desimal[i][j] * batas[j] - batas[j] + rb[j]) / (2**Nbit - 1)

############## Masih Perlu perbaikan Batas Atas###############
Datfit = []
for i in range(Individu.shape[0]):
    fitness = integrationfian(Individu[i, :])
    #print(Individu[i, :])
    Datfit.append(fitness)

if Datfit:
    fitemax = np.max(Datfit)
    nmax = np.argmax(Datfit)
############## Masih Perlu perbaikan Batas Bawah###############
Dadatfit = []
Prob = np.zeros_like(Datfit)
for generasi in range(1, Maxit+1):
    print(f"Generasi ke {generasi}")
    if generasi > 1:
        sortfit = np.random.rand(Npop, Nbit * Nvar + 1)
        Individu1 = sortfit[int((1-el)*Npop):Npop,:]
        remain = sortfit[int(el*Npop):Npop,:]
        X = Individu1
        M = X.shape[0]

        sumfitness = sum(Datfit)
        Prob = [Datfit[i]/sumfitness for i in range(M)]
        for i in range(1, M):
            Prob[i] = Prob[i] + Prob[i-1]

        Xparents = np.zeros((M, X.shape[1]))
        for i in range(M):
            n = np.random.rand()
            k = 0
            for j in range(M-1):
                if n > Prob[j].any():
                    k = j + 1
            Xparents[i,:] = X[k,:]

#crossover
        M, d = Xparents.shape
        Xcrossed = Xparents.copy()
        for i in range(0, M-1, 2):
            c = np.random.rand()
            if c <= Pc:
                p = np.ceil((d-1) * np.random.rand())
                p = int(p)
                Xcrossed[i, :] = np.concatenate((Xparents[i, 0:p], Xparents[i+1, p:d]))
                Xcrossed[i+1, :] = np.concatenate((Xparents[i+1, 0:p], Xparents[i, p:d]))

        if M % 2 != 0:
            c = np.random.rand()
            if c <= Pc:
                p = np.ceil((d-1) * np.random.rand())
                p = int(p)
                str = np.ceil((M-1) * np.random.rand())
                str = int(str)
                Xcrossed[M-1, :] = np.concatenate((Xparents[M-1, 0:p], Xparents[str, p:d]))
    
###mutasi
        Xnew = Xcrossed
        [M, d] = np.shape(Xcrossed)
        for i in range(M):
            for j in range(d):
                p = np.random.rand()
                if p <= Pm:
                    Xnew[i][j] = 1 - Xcrossed[i][j]
        print(f'New fitness calculation:{generasi}')
        Bangkit = np.concatenate((Xnew[:, :Nbit*Nvar], remain[:, :Nbit*Nvar]))
    #eBangkit = Bangkit
    for i in range(Npop):
        for j in range(Nvar):
            slice = Bangkit[i][((j*Nbit)-(Nbit-1)):(j*Nbit)]
            binary_representation = ''.join(map(lambda x: '1' if x >= 0.5 else '0', slice))
            if binary_representation:
                Desimal[i][j] = int(binary_representation, 2)
            else:
                Desimal[i][j] = 0
            Individu[i][j] = (Desimal[i][j] * batas[j] - batas[j] + rb[j]) / (2**Nbit - 1)
    
############## Masih Perlu perbaikan Batas Atas###############
    Datfit = []
    for i in range(Npop):
        fitness = integrationfian(Individu[i,:])
        Datfit.append(fitness)
    
############## Masih Perlu perbaikan Batas Bawah###############
    
    fitemax = np.max(Datfit)
    nmax = np.argmax(Datfit)
    nmax=100
    Dadatfit = Datfit
    eDadatfit.append(Dadatfit)
    eIndividu.append(Individu)
    fitnessmax = np.max(eDadatfit)
    nmax = np.argmax(eDadatfit)
    efitnessmax.append(fitnessmax)
    #BangkitMax = eBangkit[int(nmax), :]
    #IndividuMax = eIndividu[int(nmax), :]
    #eIndividuMax.append(IndividuMax)
    #BangkitMaxlast = BangkitMax
    #schedmax = BangkitMax
    #sort = [Bangkit, Dadatfit]
    #summary.append(sort)
    #david.append(Dadatfit)

#Optimum_variable_design = eIndividuMax[0, :]
#Optimum_objective_function = fitness[0, :]

#AFR = eIndividuMax[0, 8] / eIndividuMax[0, 3]


plt.title('Grafik Fitness GA selama Iterasi', color='b')
plt.xlabel('Iterasi')
plt.ylabel('Efisiensi (0-1)')
plt.plot(efitnessmax, label='efitnessmax')
plt.legend()
plt.show()

