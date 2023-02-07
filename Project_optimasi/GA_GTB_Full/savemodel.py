import tensorflow as tf
import openpyxl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

wb = openpyxl.load_workbook(r'/content/A.xlsx')
sheet = wb.active
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)
a = np.array(data)


################ Kode Sumber ##################################
ut = a[3:7000,:8]
yt = a[3:7000,8:11]
uv = a[7000:8000,:8]
yv = a[7000:8000,8:11]
us = a[3:8000,:8]
ys = a[3:8000,8:11]
us=np.array(us, dtype=np.float32)   #8x6997
uv=np.array(uv, dtype=np.float32)   #8x1000
ys=np.array(ys, dtype=np.float32)   #3x6997
yv=np.array(yv, dtype=np.float32)   #3x1000
neural(us,ys,uv,yv)