from glob import glob
from PIL import Image
import cv2
import os
import shutil

from face_detection.yolov5 import face_detect
from face_detection.yolov5.utils.general import plot_one_box
from emotion_classification import predict


def clear_dir(dir):
    if os.path.exists(dir):  # output dir
        shutil.rmtree(dir)  # delete dir
    os.makedirs(dir)  # make new dir


def yolo_str_cxcywh2xyxy(bbox, image_size):
    bbox = [float(i) for i in bbox]
    img_w, img_h = image_size
    bbox[0] = bbox[0] * img_w
    bbox[1] = bbox[1] * img_h
    bbox[2] = bbox[2] * img_w
    bbox[3] = bbox[3] * img_h
    x1 = int(bbox[0] - bbox[2] / 2)
    y1 = int(bbox[1] - bbox[3] / 2)
    x2 = int(bbox[0] + bbox[2] / 2)
    y2 = int(bbox[1] + bbox[3] / 2)
    return [x1, y1, x2, y2]


def crop_image(image, bbox, name, parameters):
    cnt = 0
    for b in bbox:
        cropped_image = image.crop(yolo_str_cxcywh2xyxy(b, image.size))
        cropped_image.save(parameters['crop_dir'] + '\\' + name.split('.')[0] + f'_{cnt}.jpg')
        cnt += 1


def main():
    # face detection
    parameters = {'weights': 'weights/face_detection_yolov5.pt',
                  'source': 'inference/images',
                  'img_size': 640,
                  'conf_thres': 0.25,
                  'iou_thres': 0.45,
                  'device': '',
                  'save_dir': 'inference/output',
                  'view_img': False,
                  'save_txt': True,
                  'crop_dir': 'inference/cropped_images'
                  }

    # clear dir
    clear_dir(parameters['save_dir'])
    clear_dir(parameters['crop_dir'])

    assert len(os.listdir(parameters['source'])) > 0, f'No files in {parameters["source"]}'
    assert len(os.listdir(parameters['save_dir'])) == 0
    assert len(os.listdir(parameters['crop_dir'])) == 0

    face_detect.detect(parameters)

    # crop
    images_path = glob(parameters['source'] + '\\*.jpg')
    bboxes_path = [p.split('\\')[-1] for p in glob(parameters['save_dir'] + '\\*.txt')]

    bboxes = []
    for image_path in images_path:
        image = Image.open(image_path)
        bbox_path = image_path.split('.')[0] + '.txt'
        if not bbox_path.split('\\')[-1] in bboxes_path:
            bboxes.append([])
            continue
        bbox_path = parameters['save_dir'] + '\\' + bbox_path.split('\\')[-1]
        with open(bbox_path) as f:
            bbox = f.read().splitlines()
        bbox = [b.split(' ')[1:-1] for b in bbox if len(b) > 0]
        name = image_path.split('\\')[-1]
        bboxes.append(bbox)
        crop_image(image, bbox, name, parameters)

    # emotion classification
    image_path = parameters['crop_dir']
    weight_path = 'weights/model_epoch_53.pt'
    batch_size = 8
    preds = predict.predict(parameters['device'], image_path, weight_path, batch_size)

    num_faces = [len(bbox) for bbox in bboxes]
    i, j = 0, 0
    cut_preds = []
    for num in num_faces:
        j += num
        cut_preds.append(preds[i:j])
        i = j

    for bbox, pred, image_path in zip(bboxes, cut_preds, images_path):
        image = cv2.imread(image_path)
        for b, p in zip(bbox, pred):
            image_size = image.shape[:-1]
            image_size = image_size[::-1]
            xyxy = yolo_str_cxcywh2xyxy(b, image_size)
            plot_one_box(xyxy, image, label=p, line_thickness=2)
        save_path = parameters['save_dir'] + '\\' + image_path.split('\\')[-1]
        cv2.imwrite(save_path, image)


if __name__ == '__main__':
    main()
