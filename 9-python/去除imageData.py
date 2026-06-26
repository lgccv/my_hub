import json
import os
import cv2
import shutil
from tqdm import tqdm

path = r'C:\Users\61082\Desktop\omoron\yz_erqi\异物'
json_file = [name for name in os.listdir(path) if name.endswith('.json')]
ok_image = []
ng_image = []
for image_name in json_file:
    print(image_name)
    labelme_file = json.load(open(os.path.join(path,image_name),'r',encoding='UTF-8'))
    labelme_file['imageData'] = None
    if len(labelme_file['shapes'])==0:
          ok_image.append(image_name.replace('.json','.jpg'))
    else:
          ng_image.append(image_name.replace('.json','.jpg'))
    with open(os.path.join(path,image_name), "w",encoding='UTF-8') as f:
        json.dump(labelme_file, f,indent=4)

print(ok_image)
print('finish')

# for name in tqdm(ok_image):
#     os.remove(os.path.join(path,name))
#     os.remove(os.path.join(path,name.replace('.jpg','.json')))

# for name in tqdm(ng_image):
#     shutil.copy2(os.path.join(r'C:\Users\61082\Desktop\omoron\xingcheng\image_format',name),os.path.join(r'C:\Users\61082\Desktop\omoron\xingcheng\ng_labelme',name))
#     shutil.copy2(os.path.join(r'C:\Users\61082\Desktop\omoron\xingcheng\image_format',name.replace('.jpg','.json')),os.path.join(r'C:\Users\61082\Desktop\omoron\xingcheng\ng_labelme',name.replace('.jpg','.json')))



