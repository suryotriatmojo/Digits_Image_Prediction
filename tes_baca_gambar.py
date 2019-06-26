import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

data_digit = load_digits()

# splitting sklearn: model selection ============================================================================================================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    data_digit['data'],         # => sebagai x
    data_digit['target'],       # => sebagai y
    test_size = 0.1             # => data test = 10%
)

# logistic regression ============================================================================================================================================
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression(solver='liblinear', multi_class='auto')

# training
model_log.fit(x_train, y_train)

# skor
print(model_log.score(x_train, y_train) * 100, '%')

# masukkan gambar ======================================================================================================================================
import matplotlib.pyplot as plt
from PIL import Image

# black/ white = 'L / 'RGBA' / 'CMYK'
size = 8, 8
gambar = Image.open('2.png').convert('L')   # masih berupa objek
gambar.thumbnail(size, Image.ANTIALIAS)
gambar = np.array(gambar)

# prediksi ======================================================================================================================================
print(model_log.predict(gambar.reshape(1,-1)))   # => reshape diperlukan karena .predict harus data 2D

# draw image, show real data, show prediction ======================================================================================================================================
fig = plt.figure(figsize = (10,5))

plt.imshow(gambar, cmap='inferno')
plt.title(
    'Prediksi = {} // Tingkat Akurasi = {} %'
    .format(
        model_log.predict(gambar.reshape(1,-1))[0],
        round(model_log.score(x_train, y_train) * 100, 2)
    )
)
plt.show()