import numpy as np
import pandas as pd
import openpyxl
import numpy as np

def integrationGTGB(x):
    # Membuka workbook
    wb = openpyxl.load_workbook(r'C:\Users\fiansyah\Documents\GitHub\Algoritma\Project_optimasi\A.xlsx')
    sheet = wb.active

    # Membaca data dari worksheet dan menyimpannya dalam bentuk array Numpy
    data = []
    for row in sheet.iter_rows(values_only=True):
        data.append(row)
    A = np.array(data)

    ut = A[3:7000,:8].T
    yt = A[3:7000,8:11].T
    uv = A[7000:8000,:8].T
    yv = A[7000:8000,8:11].T
    us = A[3:8000,:8].T
    ys = A[3:8000,8:11].T

    rowv, colv = uv.shape
    rowu, colu = ut.shape
    rowy, coly = yt.shape
    Min = -np.ones((rowu, 1))
    Max = np.ones((rowu, 1))
    MM = np.concatenate((Min, Max), axis=1)

    us = ut
    ys = yt

    maxus = np.max(us, axis=1)
    minus = np.min(us, axis=1)

    maxys = np.max(ys, axis=1)
    minys = np.min(ys, axis=1)

    minmaxus = np.concatenate((maxus, minus), axis=0)
    minmaxys = np.concatenate((maxys, minys), axis=0)

    for i in range(rowy):
        yt[i,:] = ((2/(np.max(ys[i,:])-np.min(ys[i,:])))*(yt[i,:]-np.min(ys[i,:])))-1
        yv[i,:] = ((2/(np.max(ys[i,:])-np.min(ys[i,:])))*(yv[i,:]-np.min(ys[i,:])))-1


    for i in range(rowu):
        x[i] = ((2/(maxus[i]-minus[i]))*(x[i]-minus[i]))-1

    x = x.reshape((len(x), 1))
    u1, u2, u3, u4, u5, u6, u7, u8 = x[0][0], x[1][0], x[2][0], x[3][0], x[4][0], x[5][0], x[6][0], x[7][0]

    for j in range(colv):
        for i in range(rowv):
            uv[i][j] = ((2/(maxus[i]-minus[i]))*(uv[i][j]-minus[i]))-1

    ut1, ut2, ut3, ut4, ut5, ut6, ut7, ut8 = ut[0,:], ut[1,:], ut[2,:], ut[3,:], ut[4,:], ut[5,:], ut[6,:], ut[7,:]

    yt1, yt2, yt3 = yt[0,:].reshape((len(yt[0,:]), 1)), yt[1,:].reshape((len(yt[1,:]), 1)), yt[2,:].reshape((len(yt[2,:]), 1))

    uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8 = uv[0,:], uv[1,:], uv[2,:], uv[3,:], uv[4,:], uv[5,:], uv[6,:], uv[7,:]

    yv1 = yv[0, :]
    yv2 = yv[1, :]
    yv3 = yv[2, :]
    
    if yt.any() < 0 or yt.any() > 1:
        yt = 0
    else:
        yt = yt
    
    return yt
