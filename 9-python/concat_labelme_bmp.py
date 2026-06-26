import os
import shutil
from tqdm import tqdm
import json

# labelme的路径
labelmePath = [
            r'D:\2-Project\17-haideshenfenzheng\concat_dataset\train\labelme',
            r'D:\2-Project\17-haideshenfenzheng\concat_dataset\test\labelme',
            r'D:\2-Project\17-haideshenfenzheng\concat_dataset\images_yiwu_1212',
            r'D:\2-Project\17-haideshenfenzheng\concat_dataset\IDCard_defect_20241213_coco_roi\train\labelme',
            ]
# 保存labelme的路径
save_dir = r'D:\2-Project\17-haideshenfenzheng\concat_dataset\ID_dataset\train\labelme'

image_save_path = r'D:\2-Project\17-haideshenfenzheng\concat_dataset\ID_dataset\train\images'

for path in labelmePath:
    if not os.path.exists(path):
        continue
    file_names = os.listdir(path)
    image_path = []
    json_path = []
    for name in file_names:
        if '.bmp' in name:
            image_path.append(name)
        if '.json' in name:
            json_path.append(name)
    assert len(image_path) == len(set(json_path)), 'include same file name!!!'

    # 开始移动图像
    for i,imagepath in enumerate(tqdm(image_path),start=1):
        if not os.path.exists(os.path.join(save_dir,imagepath)):
            shutil.copy2(os.path.join(path,imagepath), os.path.join(save_dir,imagepath))
            shutil.copy2(os.path.join(path, imagepath), os.path.join(image_save_path, imagepath))
            json_name = imagepath.replace('.bmp', '.json')
            if json_name in json_path:
                shutil.copy2(os.path.join(path, json_name), os.path.join(save_dir, json_name))
        else:
            new_imagepath = imagepath[0:-4]+'_'+str(i)+'.bmp'
            if os.path.exists(os.path.join(save_dir,new_imagepath)):
                print(new_imagepath)
                assert 'same name!!!'
            shutil.copy2(os.path.join(path,imagepath), os.path.join(save_dir,new_imagepath))
            shutil.copy2(os.path.join(path, imagepath),os.path.join(image_save_path, new_imagepath))
            json_name = imagepath.replace('.bmp', '.json')
            if json_name in json_path:
                new_json_name = new_imagepath.replace('.bmp', '.json')
                json_name = imagepath.replace('.bmp', '.json')
                with open(os.path.join(path, json_name), 'r', encoding='utf-8') as file:
                    data = json.load(file)

                data['imagePath'] = new_imagepath
                with open(os.path.join(save_dir, new_json_name), "w", encoding='UTF-8') as f:
                    json.dump(data, f, indent=4)

print('finish')

