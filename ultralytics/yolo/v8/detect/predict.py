# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import argparse
import time
from pathlib import Path
from collections import defaultdict, deque
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import time
import numpy as np

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

if torch.cuda.is_available():
    print("GPU is available!")
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available.")

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=2,  # max_age를 2로 설정하여 2 프레임 이후에 객체가 사라졌다고 판단
                            n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: # person
        color = (85,45,255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_boxes(img, bbox, names, object_id, identities=None, object_times=None, offset=(0, 0)):
    # Remove tracked points from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # Object ID
        id = int(identities[i]) if identities is not None else 0

        # Create new buffer for new object
        if id not in data_deque:  
            data_deque[id] = deque(maxlen=64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]

        # 시간 정보 추가
        time_on_screen = object_times.get(id, 0)
        label = f'ID {id}: {obj_name} | Time: {time_on_screen:.2f}s'

        # Add center to buffer
        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)

        # Draw trail
        for j in range(1, len(data_deque[id])):
            if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
            cv2.line(img, data_deque[id][j - 1], data_deque[id][j], color, thickness)
    return img

class DetectionPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_objects = defaultdict(lambda: {'frames_seen': 0, 'last_seen': -1})
        self.object_times = {}  # 각 객체의 화면에 나타난 시간을 저장
        self.saved_objects = set()  # 이미 저장된 객체의 ID를 추적하기 위한 집합

        # 저장 디렉토리 설정
        self.object_save_dir = Path("./saved_object")
        self.object_save_dir.mkdir(parents=True, exist_ok=True)

        # FPS 설정
        if hasattr(self.dataset, 'fps') and self.dataset.fps:
            self.fps = self.dataset.fps
        else:
            self.fps = 30  # 기본값

        # 첫 번째 프레임의 크기를 가져오기 위해 초기화
        self.frame_width = None
        self.frame_height = None
        self.video_writer = None  # VideoWriter는 첫 번째 프레임에서 초기화

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        # GPU 사용을 위해 모델 장치를 강제로 CUDA로 설정
        self.model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img = torch.from_numpy(img).to(self.model.device)  # GPU로 데이터 이동
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            # Person 클래스만 필터링
            if pred is not None and len(pred):
                pred = pred[pred[:, 5] == 0]  # 클래스 0은 'Person'을 의미합니다.
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            if pred is not None and len(pred):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            preds[i] = pred

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        # 첫 번째 프레임에서 VideoWriter 초기화
        if self.video_writer is None:
            # 프레임 크기 설정
            self.frame_height, self.frame_width = im0.shape[:2]
            # VideoWriter 초기화
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 ('mp4v', 'XVID' 등)
            video_save_path = str(self.object_save_dir / 'output_video.mp4')  # 저장할 비디오 경로
            print(f"비디오 저장 경로: {video_save_path}")
            self.video_writer = cv2.VideoWriter(video_save_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        det = preds[idx]
        all_outputs.append(det)
        if det is None or len(det) == 0:
            # 현재 프레임에 감지된 객체가 없을 경우, 기존 객체가 사라졌는지 체크
            self.check_left_objects(frame)
            # 프레임을 그대로 저장
            if self.video_writer is not None:
                self.video_writer.write(im0)
            return log_string

        # 클래스 카운트 로그
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []

        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, oids, im0)
        current_ids = []
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            current_ids = identities.tolist()

            # 각 객체의 시간 업데이트
            for i, obj_id in enumerate(identities):
                obj_id = int(obj_id)
                if obj_id in self.tracked_objects:
                    obj = self.tracked_objects[obj_id]
                    if obj["last_seen"] == frame - 1:
                        obj["frames_seen"] += 1
                    else:
                        obj["frames_seen"] += 1  # 프레임이 연속되지 않더라도 누적
                    obj["last_seen"] = frame
                else:
                    # 새로운 객체 추적 시작
                    self.tracked_objects[obj_id] = {"frames_seen": 1, "last_seen": frame}

                # 시간 계산 및 저장
                time_on_screen = self.tracked_objects[obj_id]["frames_seen"] / self.fps
                self.object_times[obj_id] = time_on_screen

                # 노출 시간이 1초를 초과하고, 아직 저장되지 않은 객체라면 이미지 저장
                if time_on_screen > 1.0 and obj_id not in self.saved_objects:
                    self.save_object_image(im0, bbox_xyxy[i], obj_id)
                    self.saved_objects.add(obj_id)

            # 객체별 시간 정보를 draw_boxes에 전달
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, self.object_times)

        # 현재 프레임에서 사라진 객체 체크
        self.check_left_objects(frame, current_ids)

        # 처리된 프레임을 비디오로 저장
        if self.video_writer is not None:
            self.video_writer.write(im0)

        # 결과를 표시하고 싶다면 아래 코드를 유지합니다.
        if self.args.show:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        return log_string

    def save_object_image(self, img, bbox, obj_id, min_size=80):
        """객체의 이미지를 바운딩 박스로 잘라 저장합니다.
        Args:
            img: 현재 프레임 이미지
            bbox: 바운딩 박스 좌표 (x1, y1, x2, y2)
            obj_id: 객체 ID
            min_size: 객체 저장을 위한 최소 크기 (픽셀)
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        w, h = x2 - x1, y2 - y1

        # 너비와 높이를 최소 크기로 확장
        if w < min_size:
            diff = (min_size - w) // 2
            x1 -= diff
            x2 += diff
        if h < min_size:
            diff = (min_size - h) // 2
            y1 -= diff
            y2 += diff

        # 이미지 경계를 넘어가지 않도록 제한
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img.shape[1])  # img.shape[1]은 이미지의 너비
        y2 = min(y2, img.shape[0])  # img.shape[0]은 이미지의 높이

        # 최종 너비와 높이를 다시 계산
        w, h = x2 - x1, y2 - y1

        # 너비와 높이가 여전히 최소 크기보다 작은 경우 추가 확장
        if w < min_size:
            x2 = min(x2 + (min_size - w), img.shape[1])
            x1 = max(x1 - (min_size - w), 0)
        if h < min_size:
            y2 = min(y2 + (min_size - h), img.shape[0])
            y1 = max(y1 - (min_size - h), 0)

        # 최종 바운딩 박스 크기 확인
        w, h = x2 - x1, y2 - y1
        assert w >= min_size and h >= min_size, "Bounding box resizing failed to meet minimum size"

        # 바운딩 박스 크기 조정 후 이미지 잘라내기
        obj_img = img[y1:y2, x1:x2]
        obj_img_path = self.object_save_dir / f"object_{obj_id}.jpg"
        cv2.imwrite(str(obj_img_path), obj_img)
        print(f"ID {obj_id} 객체의 이미지를 저장했습니다. 크기: {obj_img.shape[1]}x{obj_img.shape[0]}")

    def check_left_objects(self, current_frame, current_ids=[]):
        # 현재 추적 중인 객체 중에서 current_ids에 없는 객체를 찾습니다.
        for obj_id in list(self.tracked_objects.keys()):
            obj = self.tracked_objects[obj_id]
            if obj_id not in current_ids:
                frames_missing = current_frame - obj["last_seen"]
                time_missing = frames_missing / self.fps
                if time_missing >= 0.05:
                    # 객체가 화면에서 벗어났다고 판단
                    total_time = obj["frames_seen"] / self.fps
                    print(f"ID {obj_id} 객체가 화면에서 벗어났습니다. 총 노출 시간: {total_time:.2f}초")
                    # 객체 정보 삭제
                    del self.tracked_objects[obj_id]
                    if obj_id in self.object_times:
                        del self.object_times[obj_id]
                    if obj_id in data_deque:
                        del data_deque[obj_id]
                    if obj_id in self.saved_objects:
                        self.saved_objects.remove(obj_id)

    def on_predict_end(self):
        # 모든 예측이 끝난 후 남아있는 객체의 총 시간을 출력
        print("남아있는 객체의 화면 노출 시간:")
        for obj_id, obj_info in self.tracked_objects.items():
            total_time = obj_info["frames_seen"] / self.fps
            print(f"ID {obj_id}: {total_time:.2f}초")

        # VideoWriter 해제
        if self.video_writer is not None:
            self.video_writer.release()
            print("비디오 저장이 완료되었습니다.")

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    torch.backends.cudnn.benchmark = True  # 성능 최적화

    # 실행 시작 시간 기록
    start_time = time.time()

    init_tracker()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # 이미지 크기 확인
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

    # 실행 종료 시간 기록
    end_time = time.time()
    elapsed_time = end_time - start_time  # 전체 실행 시간 계산
    print(f"전체 실행 시간: {elapsed_time:.2f}초")  # 실행 시간 출력

if __name__ == "__main__":
    predict()
