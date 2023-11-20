from PIL import Image
import os.path

targerdir = "./train" #해당 폴더 설정

files = os.listdir(targerdir)

format = [".jpg",".png",".jpeg","bmp",".JPG",".PNG","JPEG","BMP"] #지원하는 파일 형태의 확장자들
for (path,dirs,files) in os.walk(targerdir):
    for file in files:
         if file.endswith(tuple(format)):
             image = Image.open(path+"/"+file)
             print(image.filename)
             print(image.size)

             image=image.resize((1024, 768))
             image.save('./1024x768/'+file)
             print(image.size)

         else:
             print(path)
             print("InValid",file)