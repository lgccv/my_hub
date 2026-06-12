import os
import platform
import cv2
import numpy as np
import time
from PIL import Image,ImageDraw,ImageFont
from rknn.api import RKNN

class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')            
            exit(ret)
        print('done')
        
        self.rknn = rknn


    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result

    def release(self):
        self.rknn.release()
        self.rknn = None

class yolo26Segment:
    def __init__(self,resize=(320,320),score_threshold = 0.2):
        self.resize_width, self.resize_height = resize
        self.class_name = ["cardboard_smallbox","gray_box","guiding","material_platform","tag_code_m","tray"]
        self.threshold = score_threshold
        self.model_path = "/home/std/workspace-hub/lgc/yolov26/runs/segment/train-5/weights/last.rknn"
        self.target = "rk3588"
        self.model = RKNN_model_container(self.model_path, self.target)


    def lettle_box(self,img,resize=(640,640),padding_value=114):
        '''
            letterbox的原理:
            1、首先将图像的长边缩放到640,等比例缩放后,短边的长度一定是小于640
            2、短边与640的距离差多少,除以2就是边缘填充宽度
            3、在边缘填充宽度填入114的值
        '''
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = resize
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = round(dh - 0.1) , round(dh + 0.1)
        left, right = round(dw - 0.1) , round(dw + 0.1)
        h, w, c = img.shape
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(padding_value,) * 3
        )
        return img


    def pipeline(self,inputs):
        t1 = time.time()
        images = self.preprocess(inputs)
        outputs = self.infer(images)
        result = self.postprocess(images,inputs,outputs)
        t2 = time.time()
        print("ct:",t2-t1)
        return result
        
    def preprocess(self,inputs):
        images= []
        for image in inputs:
            im = self.lettle_box(img = image,resize=(self.resize_width,self.resize_height),padding_value=114)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = np.transpose(im, [2, 0, 1])
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32)
            im /= 255
            im = np.round(im,4)
            images.append(im)
        images = np.stack(images, axis=0)  # (B, 3, H, W)
        return {"input":images}

    def infer(self,images):
        outputs = self.model.run(images["input"])
        return outputs
    

    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        boxes = np.asarray(boxes)
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)
        r = np.arange(w, dtype=boxes.dtype)[None, None, :]
        c = np.arange(h, dtype=boxes.dtype)[None, :, None]
        return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))
        x1, y1, x2, y2 = boxes.round().astype(np.int32).T
        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)
        for i in range(n):
            masks[i, :y1[i], :] = 0
            masks[i, y2[i]:, :] = 0
            masks[i, :, :x1[i]] = 0
            masks[i, :, x2[i]:] = 0
        return masks


    def scale_boxes(self,img1_shape,boxes,img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
        pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        boxes[..., 2] -= pad_x  # x padding
        boxes[..., 3] -= pad_y  # y padding
        boxes[..., :4] /= gain
        h,w = img0_shape[0],img0_shape[1]
        boxes[..., 0] = np.clip(boxes[..., 0], 0, w)  # x1
        boxes[..., 1] = np.clip(boxes[..., 1], 0, h)  # y1
        boxes[..., 2] = np.clip(boxes[..., 2], 0, w)  # x2
        boxes[..., 3] = np.clip(boxes[..., 3], 0, h)  # y2
        return boxes

    def scale_masks(self,masks,shape):
        im1_h,im1_w = masks.shape[2:]
        im0_h,im0_w = shape[:2]
        if im1_h == im0_h and im1_w ==im0_w:
            return masks
        gain = min(im1_h / im0_h, im1_w / im0_w)  # gain  = old / new
        pad_w, pad_h = (im1_w - round(im0_w * gain)), (im1_h - round(im0_h * gain))  # wh padding
        pad_w /= 2
        pad_h /= 2
        top, left = (round(pad_h - 0.1), round(pad_w - 0.1)) 
        bottom = im1_h - round(pad_h + 0.1)
        right = im1_w - round(pad_w + 0.1)
        cropped = masks[..., top:bottom, left:right].astype(np.float32)
        n, c = cropped.shape[:2]
        out_h, out_w = shape[:2]
        resized = np.empty((n, c, out_h, out_w), dtype=np.float32)
        for i in range(n):
            for j in range(c):
                resized[i, j] = cv2.resize(
                    cropped[i, j],
                    (out_w, out_h),
                    interpolation=cv2.INTER_LINEAR,
                )
        return resized

    
    def postprocess(self,images,inputs,output):
        results = []
        protos = output[1]
        preds = output[0]
        preds = [pred[pred[:, 4] > self.threshold][:300] for pred in preds]
        for pred,img,ori_image,proto in zip(preds,images["input"],inputs,protos):
            bboxes = pred[:,:4]
            masks_in = pred[:,6:]
            shape = img.shape[1:]
            c, mh, mw = proto.shape
            masks = (masks_in @ proto.astype(np.float32).reshape(c, -1)).reshape(-1, mh, mw)
            width_ratio = mw /shape[1]
            height_ratio = mh /shape[0]

            ratios = np.array([[width_ratio, height_ratio, width_ratio, height_ratio]], dtype=np.float32)

            masks = self.crop_mask(masks,bboxes*ratios)
            masks = np.stack([cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)for mask in masks], axis=0) 
            masks = (masks > 0.0).astype(np.uint8)
            masks = self.scale_masks(masks[None].astype(np.float32),(ori_image.shape[0],ori_image.shape[1]))[0] > 0.5

            if masks is not None:
                keep = masks.max(axis=(-2,-1)) >0
                if not np.all(keep):
                    pred,masks = pred[keep],masks[keep]

            pred[:,:4] = self.scale_boxes(img.shape[1:],pred[:,:4],ori_image.shape)

            result = []
            for ind,(det,mask) in enumerate(zip(pred[:,:6],masks)):
                mask_img = np.zeros(ori_image.shape[:2], dtype=np.uint8)
                mask_img[mask.astype(bool)] = 255
                result.append([float(det[0]),float(det[1]),float(det[2]),float(det[3]),float(det[4]),self.class_name[int(det[5])],mask_img])
            results.append(result)
        return results

            
if __name__ == "__main__":
    model = yolo26Segment(resize=(320,320),score_threshold = 0.25)
    images_dir = "/home/std/workspace-hub/lgc/yolov26/test_images"
    image_files = os.listdir(images_dir)

    batch_size = 1

    num = len(image_files)
    index = 0
    while index < num:
        index +=batch_size

        if index <= num:
            batch_image_file = image_files[index-batch_size:index]
        else:
            batch_image_file = image_files[index-batch_size:num]

        inputs = []
        for image_file in batch_image_file:
            image_file = "1770379733_078880000_1000111_png.rf.t72mveomHYqClik1DLuu.png"
            image_path = os.path.join(images_dir,image_file)
            image = cv2.imread(image_path)
            inputs.append(image)

        results = model.pipeline(inputs)

        for image_file,image,result in zip(batch_image_file,inputs,results):
            for x0,y0,x1,y1,score,label_name,mask in result:
                s = f'{label_name} | {score:.2f}'
                alpha = 0.45
                overlay = image.copy()
                mask_bool = mask.astype(bool)
                overlay[mask_bool] = (0, 255, 0)  # BGR 绿色
                image = cv2.addWeighted(overlay,alpha,image,1-alpha,0)

                cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0,0,255), 2)
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image_pil)
                fontStyle = ImageFont.truetype('/home/std/workspace-hub/lgc/yolov26/deploy/simsun.ttc', 25, encoding="utf-8")
                draw.text((int(x0), int(y0)), s, (255, 255, 255), font=fontStyle)
                image = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join("/home/std/workspace-hub/lgc/yolov26/result",image_file),image)






        
