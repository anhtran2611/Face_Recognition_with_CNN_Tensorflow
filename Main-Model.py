import numpy as np
import os
from PIL import Image
from keras import layers
from keras import models
import cv2
# TRAIN_DATA = 'datasets/train-data'
# TEST_DATA = 'datasets/test-data'
#
# Xtrain = []
# ytrain = []
#
# Xtest = []
# ytest = []
#
# dic = {'BaoAnh': [1, 0], 'Duy': [0, 1], 'BaoAnhTest': [1, 0], 'DuyTest': [0, 1]}
#
# def getData(dirData, lstData):
#     for whatever in os.listdir(dirData):
#         whatever_path = os.path.join(dirData, whatever)
#         list_filename_path = []
#         for filename in os.listdir(whatever_path):
#             filename_path = os.path.join(whatever_path, filename)
#             label = filename_path.split('\\')[1]
#             img = np.array(Image.open(filename_path))
#             list_filename_path.append((img, dic[label]))
#
#         lstData.extend(list_filename_path)
#     return lstData
#
# Xtrain = getData(TRAIN_DATA, Xtrain)
# Xtest = getData(TEST_DATA, Xtest)
#
# # np.random.shuffle(Xtrain)
#
# #Build model
# models_training_first = models.Sequential([   # Tạo ra dạng sequential : chuỗi
#     layers.Conv2D(32, (3,3), input_shape=(128, 128, 3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#
#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#
#     layers.Flatten(),
#     layers.Dense(3000, activation='relu'),  #3000 neurons , các neuron sẽ kết nối với tất cả các đơn vị của Flatten
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(2, activation='softmax') # Lớp đầu ra của model được sử dụng để phân loại ảnh thành 2 lớp
# ])
#
#
#
# # Compile model
# models_training_first.compile(optimizer='adam',
#                               loss='categorical_crossentropy',
#                               metrics=['accuracy'])
#
# # Chuyển danh sách các cặp (ảnh, one-hot encoded labels) thành các mảng NumPy
# X_train_images, y_train_labels = zip(*Xtrain)
# X_train_images = np.array(X_train_images)
# y_train_labels = np.array(y_train_labels)
#
# # Huấn luyện mô hình
# models_training_first.fit(X_train_images, y_train_labels, epochs=10, batch_size=32)
# # -	`batch_size`: là số lượng mẫu dữ liệu được sử dụng để cập nhật trọng số trong một lần huấn luyện. Ở đây, `batch_size=32` có nghĩa là mỗi lần huấn luyện, mô hình sẽ sử dụng 32 mẫu dữ liệu để cập nhật trọng số.
# # Lưu lại mô hình
models_training_first.save('model-nhom.keras')

# Đọc model
models = models.load_model('model-nhom.keras')
lstResult = ['Duy', 'Bao Anh']
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml') # bộ detect

camera = cv2.VideoCapture(1) # Open your camera
while True:
    OK, frame = camera.read()

    faces = face_detector.detectMultiScale(frame, 1.3, 5) # sd bộ detection

    for(x,y,w,h) in faces:
        roi = cv2.resize(frame[y: y+h, x: x+w],(128,128))
        result =np.argmax(models.predict(roi.reshape((-1,128,128,3))))
        cv2.rectangle(frame,(x,y),(x+w, y+h),(128, 255, 50),1) # vẽ khung hình
        cv2.putText(frame, lstResult[result], (x + 15, y-15),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 25, 255),2)

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()





















