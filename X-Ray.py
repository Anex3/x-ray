import base64
import numpy as np
import cv2
import torch
from numpy import random
from flask import Flask, request, jsonify
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.plots import plot_one_box, plot_one_box_PIL
from utils.torch_utils import select_device  # 包含提示信息

allow_address = []
allow_number = 1


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


weights = r'weights/best.pt'
opt_device = ''  # device = 'cpu' or '0' or '0,1,2,3'
imgsz = 640
opt_conf_thres = 0.25
opt_iou_thres = 0.5
# Initialize
set_logging()
device = select_device(opt_device)
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


def predict_path(path):
    # Run inference
    ret = {}  # 空字典
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # base64解码->转为np数组->opencv读取
    database = path.split(',')[1]
    data64 = base64.b64decode(database)  # base64解码得到图片数据
    img_array = np.frombuffer(data64, dtype=np.uint8)  # 将data64以流的形式读入转化成nparray对象
    im0s = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)  # 从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;主要用于从网络传输数据中恢复出图像。BGR
    im0s = im0s.astype(np.uint8)  # 将浮点数转换为整数，小数部分会被截断
    im0s = cv2.applyColorMap(im0s, cv2.COLORMAP_JET)  # cv2转化为热力图，cv2.COLORMAP_JET表示转换成热力图
    img = letterbox(im0s, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # pred = model(img, augment=opt.augment)[0]
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)
    # Process detections

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            data_tuple = []
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                prob = round(float(conf) * 100, 2)  # round 2
                position = {'x0': int(xyxy[0]), 'y0': int(xyxy[1]), 'x1': int(xyxy[2]), 'y1': int(xyxy[3])}
                data_dict = {"region": position, 'article': label, 'confidence': str(prob)}
                data_tuple.append(data_dict)
                plot_one_box(xyxy, im0s, label=f'{names[int(cls)]}', color=colors[int(cls)], line_thickness=3)
                # im0s = plot_one_box_PIL(xyxy, im0s, label=f'{names[int(cls)]}',
                #                        color=colors[int(cls)], line_thickness=None)
            # 将数据流图片编码成base64准备输出
            retval, buffer = cv2.imencode('.png', im0s[..., ::-1])
            base64_data = base64.b64encode(buffer)
            s = base64_data.decode()
            a = 'data:image/png;base64,'
            data64 = str(a + s)
            # 将数据传入字典中准备输出
            xml_head = {'x-image-tagged': data64}
            ret.update(xml_head)
            xml_body = {'recognizations': data_tuple}
            ret.update(xml_body)
    return ret


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/x/article/recognize', methods=["POST"])
def json():
    database = {}
    if request.method == 'POST':
        address = request.remote_addr
        data = request.get_data()
        data = data.decode()
        # data = data.split('"', 4)[3]
        if address in allow_address:
            database = predict_path(data)
            print('检测完成')
        else:
            if len(allow_address) < allow_number:
                allow_address.append(address)
                database = predict_path(data)
                print('检测完成')
            else:
                print('抱歉 IP %s ，您未注册使用本服务，请联系工作人员' % address)
    return jsonify(database)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)
