from ultralytics import YOLO
import numpy as np
import cv2
from typing import Any
import onnxruntime as ort
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def infer_by_onnx_yolov11():
    # Load a pretrained YOLO11n model
    model = YOLO("/home/std/workspace-hub/lgc/yolov11/runs/detect/person_detect/weights/last.onnx")
    # model = YOLO("/home/standard/yolov11/ultralytics/yolo11n.pt")
    # Perform object detection on an image
    image_path = r"/home/std/workspace-hub/lgc/yolov11/images/images_0401"
    output_path =r"/home/std/workspace-hub/lgc/yolov11/result/onnx_normal"


    rectangle_count =0 
    for name in tqdm(sorted(os.listdir(image_path),key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))):
        # name = "1774939901_500253952_134.jpg"
        results = model(os.path.join(image_path,name),conf = 0.25,iou = 0.45)  # Predict on an image
        results[0].show()  # Display results
        results[0].save(filename=os.path.join(output_path,name))
        rectangle_count = rectangle_count + len(results[0].boxes.data)
        # exit()

    print("检测到总框数:",rectangle_count)


class LetterBox:
    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, labels: dict[str, Any] | None = None, image: np.ndarray = None) -> dict[str, Any] | np.ndarray:
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]

        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
            )
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    @staticmethod
    def _update_labels(labels: dict[str, Any], ratio: tuple[float, float], padw: float, padh: float) -> dict[str, Any]:
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels
    

class DefectModel:
    def __init__(self,resize = (640,640),score_threshold=0.25, iou_threshold=0.7):
        # 初始化
        self.data_type = 'fp32'
        self.letterbox = LetterBox(resize,auto=False,stride=32)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.resize = resize
        self.class_name = ['person','bicycle','car','motorcycle','airplane','bus','train']
        self.init_model()

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

    def init_model(self, device_type='cuda', device_id=0):
        # 加载模型
        if device_type == 'cuda':
            session_options = ort.SessionOptions()
            providers = ['CPUExecutionProvider']
            options = [{}]
            is_cuda_available = ort.get_device() == 'GPU'
            if is_cuda_available:
                providers.insert(0, 'CUDAExecutionProvider')
                options.insert(0, {'device_id': device_id})
            # 用GPU推理，用哪张GPU推理
            self.session = ort.InferenceSession(os.path.join("/home/standard/yolov11/ultralytics/runs/detect/test_coco_yolo8/weights/last.onnx"), session_options, providers=providers, provider_options=options)
            self.input_name = self.session.get_inputs()[0].name   # 模型的输入名字
            self.output_name = self.session.get_outputs()[0].name # 模型的输出名字
            self.input_shape = self.session.get_inputs()[0].shape # 输入图片的shape
            self.is_cuda_available = is_cuda_available

    def preprocess(self, im):
        # 首先经过letterbox
        # im = self.letterbox(image = im)
        im = self.lettle_box(img = im,resize=(640,640),padding_value=114)
        # 将BGR转换为RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # 将BHWC转换为 BCHW
        im = np.transpose(im, [2, 0, 1])
        #保证数据连续
        im = np.ascontiguousarray(im)  # contiguous
        # 将数据转为fp16或者fp32
        im = im.astype(np.float32)
        # 数据归一化处理
        im /= 255
        return im
    
    def infer(self,input_data):
        # 推理代码
        input_data = input_data[None, :, :, :]
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs
    
    def xywh2xyxy(self,x):
        y = np.empty_like(x)  # 创建相同形状的空数组，比clone/copy更快
        
        # 提取中心点坐标和半宽高
        xy = x[..., :2]  # 中心点坐标 [x_center, y_center]
        wh = x[..., 2:4] / 2  # 半宽高 [width/2, height/2]
        
        # 计算左上角和右下角坐标
        y[..., :2] = xy - wh  # 左上角 [x1, y1]
        y[..., 2:4] = xy + wh  # 右下角 [x2, y2]
        return y
    
    def calculate_iou(self,box1, boxes):
        """
        计算一个边界框与多个边界框的IoU
        """
        # 计算交集区域的坐标
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        # 计算交集面积
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算各自面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 计算并集面积
        union = area1 + area2 - intersection
        
        # 避免除零错误
        union = np.maximum(union, 1e-8)
        
        # 计算IoU
        iou = intersection / union
        return iou

    def nms(self,detections, iou_threshold=0.5):
        if len(detections) == 0:
            return np.array([], dtype=int)
        
        # 提取坐标和分数
        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        # 按置信度分数降序排序
        sorted_indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(sorted_indices) > 0:
            # 选取当前最高分的检测框
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # 获取当前框和剩余框
            current_box = boxes[current_idx]
            remaining_indices = sorted_indices[1:]
            remaining_boxes = boxes[remaining_indices]
            
            # 计算当前框与剩余所有框的IoU
            ious = self.calculate_iou(current_box, remaining_boxes)
            
            # 保留IoU小于阈值的框（移除重叠度高的框）
            low_overlap_mask = ious <= iou_threshold
            sorted_indices = remaining_indices[low_overlap_mask]
        
        return np.array(keep)
    
    def class_aware_nms(self,detections, iou_threshold=0.5):
        """
        按类别分别进行NMS
        """
        if len(detections) == 0:
            return np.array([])
        
        # 获取所有类别
        unique_classes = np.unique(detections[:, 5])
        final_keep = []
        
        for class_id in unique_classes:
            # 获取当前类别的所有检测
            class_mask = detections[:, 5] == class_id
            class_detections = detections[class_mask]
            
            if len(class_detections) == 0:
                continue
            
            # 对当前类别执行NMS
            class_keep_indices = self.nms(class_detections, iou_threshold)
            
            # 将类别内的索引转换为全局索引
            global_indices = np.where(class_mask)[0][class_keep_indices]
            final_keep.extend(global_indices)
        
        return np.array(final_keep)
    
    def clip_boxes(self,boxes, shape):
        h, w = shape[:2]  # supports both HWC or HW shapes
        boxes[..., 0].clamp_(0, w)  # x1
        boxes[..., 1].clamp_(0, h)  # y1
        boxes[..., 2].clamp_(0, w)  # x2
        boxes[..., 3].clamp_(0, h)  # y2
        return boxes
    
    def scale_boxes(self,img1_shape, boxes, img0_shape):
        """"
        将letterbox上的坐标转换成原图的坐标
        1、首先将坐标减去坐上左上顶点
        2、然后将坐标除以缩放比例
        """
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        boxes[..., 2] -= pad_x  # x padding
        boxes[..., 3] -= pad_y  # y padding
        boxes[..., :4] /= gain
        return boxes

    def postprocess(self,outputs,image,pre_image):
        # 后处理代码
        results = []
        prediction = outputs[0]
        prediction = np.transpose(prediction, (0, 2, 1))
        predictions = prediction.squeeze(0)
        boxes = predictions[:, :4]
        boxes = self.xywh2xyxy(boxes)
        scores = predictions[:, 4:]
        max_scores = np.max(scores, axis=1)            # 每个框的最大分数
        max_score_indices = np.argmax(scores, axis=1)  # 每个框最大分数的类别索引
        det = np.column_stack([boxes, max_scores, max_score_indices])
        # 1.先过滤分数低的[位置 4 + socre + index]
        filtered_detections = det[det[:, 4] > 0.25]
        # 2.按分数大小排序
        filtered_detections = filtered_detections[np.argsort(filtered_detections[:, 4])[::-1]]
        # 3、NMS
        keep_idxs = self.class_aware_nms(filtered_detections,self.iou_threshold)
        final_detections = filtered_detections[keep_idxs]
        print(f"NMS后保留: {len(final_detections)} 个检测")
        # 4、还原
        converted_boxes = self.scale_boxes(self.resize,final_detections[:,:4],image.shape)
        final_detections[:, :4] = converted_boxes
        # 5、显示
        bboxes = [] 
        for ind, det in enumerate(final_detections):
            # 计算熵
            det = det.tolist()
            index = int(det[5])
            bboxes.append([max(det[0], 0),
                            max(det[1], 0),
                            min(det[2], image.shape[1]-1),
                            min(det[3], image.shape[0]-1),
                            det[4],
                            self.class_name[index],
                            index])
                
        results.append({'bboxes': bboxes})
        return results


if __name__ == '__main__':
    infer_by_onnx_yolov11()
    # model = DefectModel(resize=(640, 640), score_threshold=0.2, iou_threshold=0.25)
    
    # image_path = r"/home/standard/yolov11/ultralytics/bus.jpg"
    # file_bytes = np.fromfile(image_path, np.uint8)
    # image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

    # # 前处理
    # pre_image = model.preprocess(image)
    # outputs = model.infer(pre_image)
    # results = model.postprocess(outputs,image,pre_image)
    # print(results)

    # for x0, y0, x1, y1, score, label_name, label_id  in results[0]['bboxes']:
    #     s = f'{label_name} | {score:.2f}'
    #     cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0,0,255), 2)
    #     image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(image)
    #     fontStyle = ImageFont.truetype('/home/standard/yolov11/ultralytics/simsun.ttc', 25, encoding="utf-8")
    #     draw.text( (int(x0), int(y0) - 10), s, (255, 255, 255), font=fontStyle)
    #     image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        
    # cv2.imwrite(os.path.join(r'/home/standard/yolov11/ultralytics/infer.jpg'), image)

