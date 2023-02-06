import scipy.io
import openpyxl

# Membaca file .mat
mat = scipy.io.loadmat('WTGTGB.mat')
hasil = mat.keys()
print(hasil)