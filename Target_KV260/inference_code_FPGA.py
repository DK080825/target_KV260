import numpy as np
import cv2
import vart
import xir
import argparse
import os
import time
import math
import log

def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))

def non_max_suppression_numpy(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5
    output = [np.zeros((0, 6), dtype=np.float32)] * bs

    for img_i in range(bs):
        x = prediction[img_i]
        x = x[x[:, 4] > conf_thres]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        boxes = xywh2xyxy_numpy(x[:, :4])
        scores = x[:, 5:]

        if multi_label:
            i, j = np.where(scores > conf_thres)
            if len(i):
                x = np.concatenate([boxes[i], scores[i, j][:, None], j[:, None].astype(np.float32)], 1)
            else:
                continue
        else:
            cls_ids = scores.argmax(1)
            cls_conf = scores.max(1)
            m = cls_conf > conf_thres
            x = np.concatenate([boxes[m], cls_conf[m, None], cls_ids[m, None].astype(np.float32)], 1)

        if not x.shape[0]:
            continue

        if classes is not None:
            x = x[np.isin(x[:, 5].astype(int), classes)]
            if not x.shape[0]:
                continue

        x = x[x[:, 4].argsort()[::-1]]
        if not agnostic:
            offsets = x[:, 5] * 7680
            b = x[:, :4] + offsets[:, None]
        else:
            b = x[:, :4]

        keep = []
        while x.shape[0]:
            keep.append(x[0])
            if len(keep) >= max_det:
                break
            iou = box_iou_numpy(b[0:1], b[1:]).reshape(-1)
            m = iou <= iou_thres
            x = x[1:][m]
            b = b[1:][m]

        output[img_i] = np.stack(keep) if len(keep) else np.zeros((0, 6))
    return output

def xywh2xyxy_numpy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def box_iou_numpy(a, b):
    N = a.shape[0]
    M = b.shape[0]
    inter = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        x1 = np.maximum(a[i, 0], b[:, 0])
        y1 = np.maximum(a[i, 1], b[:, 1])
        x2 = np.minimum(a[i, 2], b[:, 2])
        y2 = np.minimum(a[i, 3], b[:, 3])
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        inter[i] = w * h
    area1 = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area1[:, None] + area2 - inter + 1e-9)

class PostProcessNumpy:
    def __init__(self):
        self.nc = 1; self.no = self.nc + 5; self.nl = 3; self.na = 3
        self.strides = np.array([8., 16., 32.])
        self.anchors = np.array([
            [[10,13], [16,30], [33,23]], 
            [[30,61], [62,45], [59,119]], 
            [[116,90], [156,198], [373,326]]
        ])
    def __call__(self, x): 
        z = [] 
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
            grid = np.stack((grid_x, grid_y), 2).reshape(1, 1, ny, nx, 2)
            anchor_grid = (self.anchors[i] * self.strides[i]).reshape(1, self.na, 1, 1, 2)
            y = sigmoid_numpy(x[i])
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid) * self.strides[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid
            z.append(y.reshape(bs, -1, self.no))
        return np.concatenate(z, axis=1)

def make_grid_numpy(nx, ny):
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    return np.stack((x, y), axis=-1).reshape(1,1,ny,nx,2)

def preprocess_image(img, target_shape):
    h, w, _ = img.shape
    th, tw = target_shape
    scale = min(th / h, tw / w)
    
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))

    img_padded = np.full(shape=[th, tw, 3], fill_value=114, dtype=np.uint8)
    dw, dh = (tw - nw) // 2, (th - nh) // 2
    img_padded[dh:nh+dh, dw:nw+dw, :] = img_resized

    image_data = img_padded.astype(np.float32) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    
    return image_data

def scale_coords_numpy(img1_shape, coords, img0_shape):
    h0, w0 = img0_shape[:2]; h1, w1 = img1_shape[:2]
    gain_w = w0 / w1; gain_h = h0 / h1
    coords[:, [0, 2]] *= gain_w; coords[:, [1, 3]] *= gain_h
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, w0)
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, h0)
    return coords

def plot_one_box_numpy(x, img, color=None, label=None, line_thickness=3):
    """Plots one bounding box on image img."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def run_dpu_inference(runner, frame_to_process, input_height, input_width, post_processor):
    preprocessed_image = preprocess_image(frame_to_process, (input_height, input_width))
    input_data = [preprocessed_image]
    output_data = [np.empty(tuple(t.dims), dtype=np.float32) for t in runner.get_output_tensors()]
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)
    outputs_for_post = [np.transpose(raw, (0, 3, 1, 2)) for raw in output_data]
    predictions = post_processor(outputs_for_post)
    return predictions

def run_inference(xmodel_path, source_path, output_path, conf_threshold = 0.25, frame_skip=1, color_mode='rgb'):
    print("Loading XMODEL and creating DPU runner...")
    graph = xir.Graph.deserialize(xmodel_path)
    subgraph = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = next((s for s in subgraph if "DPU" in s.get_attr("device")), None)
    if dpu_subgraph is None:
        raise RuntimeError("No DPU subgraph found in the model.")
    
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    input_tensors = runner.get_input_tensors()
    input_height = input_tensors[0].dims[1]
    input_width = input_tensors[0].dims[2]
    post_processor = PostProcessNumpy()

    # Determine source type
    is_webcam = source_path.isdigit()
    is_video  = any(source_path.lower().endswith(x) for x in ['.mp4','.avi','.mov','.mkv'])
    is_image  = any(source_path.lower().endswith(x) for x in ['.jpg','.jpeg','.png'])

    # For webcam
    if is_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            del runner
            return

        frame_count = 0
        total_fps = 0
        last_known_detections = []

        while cap.isOpened():
            ret, original_frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            
            loop_start_time = time.time()
            run_detection_this_frame = (frame_count % frame_skip == 0)

            if run_detection_this_frame:
                frame_to_process = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB) if color_mode == 'rgb' else original_frame
                mode_str = "RGB" if color_mode == 'rgb' else "BGR"

                predictions = run_dpu_inference(runner, frame_to_process, input_height, input_width, post_processor)
                max_obj_conf = np.max(predictions[..., 4])
                detections = non_max_suppression_numpy(predictions, conf_thres=conf_threshold, iou_thres=0.45)[0]

                scaled_detections = []
                detections_to_log = None

                if detections is not None and len(detections):
                    original_scores = detections[:, 4]
                    multiplied_scores = np.minimum(original_scores * 1.0, 1.0)

                    valid_mask = (multiplied_scores > 0.1)

                    valid_detections = detections[valid_mask]
                    valid_multiplied_scores = multiplied_scores[valid_mask]

                    detections_to_log = valid_detections

                    if len(valid_detections):
                        scaled_coords = scale_coords_numpy((input_height, input_width), valid_detections[:, :4], original_frame.shape[:2]).round()
                        for i, (*xyxy, conf, cls) in enumerate(valid_detections):
                            scaled_detections.append((scaled_coords[i], valid_multiplied_scores[i], cls))

                log.print_frame_log(frame_count + 1, 0, mode_str,
                                    max_obj_conf, detections_to_log, 0.1)
                last_known_detections = scaled_detections

            if len(last_known_detections) > 0:
                for (xyxy, conf_x, cls) in last_known_detections:
                    label = f'Fire {conf_x:.2f}'
                    plot_one_box_numpy(xyxy, original_frame, label=label, color=[0, 0, 255])

            loop_end_time = time.time()
            frame_fps = 1 / (loop_end_time - loop_start_time + 1e-6)
            total_fps += frame_fps
            cv2.putText(original_frame, f"FPS: {frame_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Camera Feed", original_frame)  # Display the frame with detections

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        del runner
    if is_image:
        original_image = cv2.imread(source_path)
        if original_image is None:
            print(f"Error: Could not read image {source_path}.")
            del runner
            return
        frame_to_process = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        predictions = run_dpu_inference(runner, frame_to_process, input_height, input_width, post_processor)
        detections = non_max_suppression_numpy(predictions, conf_thres=conf_threshold, iou_thres=0.45)[0]
        
        if detections is not None and len(detections):
            print(f"Found {len(detections)} objects.")
            scaled_detections = scale_coords_numpy(
                (input_height, input_width), detections[:, :4], original_image.shape[:2]
            ).round()
            
            for i, (*xyxy, conf, cls) in enumerate(detections):
                label = f'Class {int(cls)} {conf:.2f}'
                plot_one_box_numpy(scaled_detections[i], original_image, label=label, color=[0, 0, 255])
        else:
            print("Found 0 objects after filtering.")
                
        output_filename = "output.jpg"
        cv2.imwrite(output_filename, original_image)
        print(f"Output image saved as {output_filename}")

        del runner

    if is_video: 
        video_path = source_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error: Could not open {video_path}"); del runner; return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: print("Warning: Video FPS invalid"); fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        LOG_THRESHOLD = 0.1 

        log.print_header(video_path, frame_width, frame_height, fps, total_frames, 
                        input_width, input_height, output_path, LOG_THRESHOLD, 
                        frame_skip, color_mode)

        frame_count = 0; total_fps = 0; last_known_detections = []
        
        while cap.isOpened():
            ret, original_frame = cap.read()
            if not ret: print("\nEnd of video file reached."); break
            
            loop_start_time = time.time()
            run_detection_this_frame = (frame_count % frame_skip == 0)
            if run_detection_this_frame:
                frame_to_process = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB) if color_mode == 'rgb' else original_frame
                mode_str = "RGB"
                predictions = run_dpu_inference(runner, frame_to_process, input_height, input_width, post_processor)
                max_obj_conf = np.max(predictions[..., 4])
                detections = non_max_suppression_numpy(predictions, conf_thres=conf_threshold, iou_thres=0.45)[0]
                
                log.print_frame_log(frame_count+1, total_frames, mode_str, 
                                    max_obj_conf, detections, LOG_THRESHOLD) 
                last_known_detections = [] 
                if detections is not None and len(detections):
                    scaled_coords = scale_coords_numpy((input_height, input_width), detections[:, :4], original_frame.shape[:2]).round()
                    for i, (*xyxy, conf, cls) in enumerate(detections):
                        last_known_detections.append((scaled_coords[i], conf, cls))
            
            if len(last_known_detections) > 0:
                for (xyxy, conf, cls) in last_known_detections:
                    label = 'Fire' 
                    plot_one_box_numpy(xyxy, original_frame, label=label, color=[0, 0, 255])
        
            loop_end_time = time.time()
            frame_fps = 1 / (loop_end_time - loop_start_time + 1e-6)
            total_fps += frame_fps
            cv2.putText(original_frame, f"FPS: {frame_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(original_frame)
            frame_count += 1
        
        log.print_summary(frame_count, total_fps, output_path)
        cap.release(); out.release(); del runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("--xmodel", required=True, type=str, help="Path to the XMODEL file.")
    required.add_argument("--source", required=True, type=str, help="Path to input source (image, video file, or webcam index).")
    optional.add_argument("--conf_threshold", default=0.25, type=float, help="Confidence threshold for detections.")
    optional.add_argument("--frame_skip", default=1, type=int, help="Number of frames to skip between detections.")
    optional.add_argument("--color_mode", default='rgb', type=str, choices=['rgb', 'bgr'], help="Color mode for input frames.")
    optional.add_argument("--output", default="output.mp4", type=str, help="Path to save output video (for video source).")
    args = parser.parse_args()

    if not os.path.isfile(args.xmodel): print(f"Error: Model file not found at {args.xmodel}"); exit()
    if (not args.source.isdigit()) and (not os.path.isfile(args.source)):
        print(f"Error: source not found at {args.source}")
        exit()

        
    run_inference(args.xmodel, args.source,  args.output, args.conf_threshold, args.frame_skip, args.color_mode)



