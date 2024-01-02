import numpy as np
import os
from PIL import Image

TRAIN_DATA = 'datasets/train-data'


Xtrain = []
ytrain = []
dict = {'BaoAnh': [1,0], 'Duy': [0,1]}
# Lấy hình trong thư mục
# os.listdir : liệt kê tất cả các phần tử trong thư mục đưa vào

for whatever in os.listdir(TRAIN_DATA):
    whatever_path = os.path.join(TRAIN_DATA,whatever) # đường dẫn đi đến thư mục đích(whatever)
    list_filename_path = []
    for filename in os.listdir(whatever_path): # đi vào từng hình ảnh
        filename_path = os.path.join(whatever_path,filename) # đường dẫn đi đến từng hình
        #print(filename_path)
        label = filename_path.split('\\')[1]
    #print(whatever_path)
        img = np.array(Image.open(filename_path)) # Đọc hình ảnh sang ma trận 3 chiều
        list_filename_path.append((img, dict[label])) #Đưa tất cả ảnh và nhãn vô trong list

    Xtrain.extend(list_filename_path) # Đưa từng hình ảnh vào Xtrain từ thư mục list_filename_path

print(Xtrain[10])

# Xây dựng nhãn ytrain cho các hình Xtrain
# Với từng mỗi ma trận hình thì có 1 label
# Tạo ra 1 dictionary để chuyển từ label chữ sang one-hot-coding

