import numpy as np
import tensorflow as ts
import keras as kr

(train_x, train_label), (test_x, test_label) = kr.datasets.cifar100.load_data()


#Normalize

nor_train_x = train_x / 255
nor_test_x = test_x / 255


#create model

model = kr.models.Sequential([
    # Convolution find Texture of image 32 filter
    kr.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    # Decrease image size
    kr.layers.MaxPooling2D((2, 2)),
    # Convolution find Texture of image 62 filter
    kr.layers.Conv2D(64, (3, 3), activation='relu'),
    kr.layers.MaxPooling2D((2, 2)),
    kr.layers.Flatten(),
    kr.layers.Dense(128, activation='relu'),
    kr.layers.Dense(100, activation='softmax'),
])

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# train model
model.fit(
    nor_train_x,
    train_label,
    epochs=20
)

model.save('model.keras')



predict_all = model.predict(np.expand_dims(nor_test_x[0], axis=0))

predicted = np.argmax(predict_all[0])

print(predicted)