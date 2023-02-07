import tensorflow as tf
import openpyxl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

def neural(us,ys,uv,yv):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(8, )),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # Train the model on the training data
    model.fit(us, ys, epochs=10, validation_data=(uv, yv))

    # Use the trained model to make predictions
    model.save("/content/hasil_predict.h5")

def integrationGTGB(a,x,yhat):
    # Define input and output matrices
    ut = a[3:7000, :8].T
    yt = a[3:7000, 8:11].T

    uv = a[7000:8000, :8].T
    yv = a[7000:8000, 8:11].T

    us = a[3:8000, :8].T
    ys = a[3:8000, 8:11].T

    # Normalize input data
    maxus = np.max(us, axis=1)
    minus = np.min(us, axis=1)
    
    max_ys = np.max(ys, axis=1)
    min_ys = np.min(ys, axis=1)

    rowv, colv = uv.shape
    rowu, colu = ut.shape
    rowy, coly = yt.shape

    for i in range(rowy):
        yt[i,:] = ((2/(np.max(ys[i,:])-np.min(ys[i,:])))*(yt[i,:]-np.min(ys[i,:])))-1
        yv[i,:] = ((2/(np.max(ys[i,:])-np.min(ys[i,:])))*(yv[i,:]-np.min(ys[i,:])))-1


    for i in range(rowu):
        x[i] = ((2/(maxus[i]-minus[i]))*(x[i]-minus[i]))-1

    x = x.reshape((len(x), 1))

    # Initialize yhat
    

    # De-normalize output
    yt = (max_ys[1] - min_ys[1]) * (yhat[1] + 1) / 2 + min_ys[1]
    if yt < 0 or yt > 1:
        yt = 0
    else:
        yt = yt

    return yt

def integrationfian(x):
    ################ Kode Sumber ##################################
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
    u = np.array(x, dtype=np.float32)   #8x1
    u = u.reshape(1, -1) # reshape u to have a shape of (1, 8)
    us=np.array(us, dtype=np.float32)   #8x6997
    uv=np.array(uv, dtype=np.float32)   #8x1000
    ys=np.array(ys, dtype=np.float32)   #3x6997
    yv=np.array(yv, dtype=np.float32)   #3x1000
    model = tf.keras.models.load_model('/content/hasil_predict.h5')
    yhat1 = model.predict(u)
    yhat=yhat1[0]
    hasil = integrationGTGB(a,x,yhat)
    return hasil