from PIL import Image
import os

input_folder = "in/"
output_folder = "out/"
size = (128, 128)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".JPG"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        resized_img = img.resize(size)
        output_path = os.path.join(output_folder, f"resized_{filename}")
        resized_img.save(output_path)