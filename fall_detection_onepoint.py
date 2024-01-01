import cv2
import time
import torch
import argparse
import numpy as np
import csv
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt
from trainer import Get_coord, draw_border
from PIL import ImageFont, ImageDraw, Image

# Initialize the CSV file to store the data
csv_filename = 'fall_detection_output.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Writing headers for keypoints and bounding box coordinates, and fall detection status
    writer.writerow(
        ['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'kpt_x', 'kpt_y', 'kpt_conf', 'bbox_kpt_x_diff', 'bbox_kpt_y_diff', 'fall_detected'])

@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu'):
    video_path = source
    device = select_device(device)
    model = attempt_load(poseweights, map_location=device)  # load FP32 model
    model.eval()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')

    # Get the original video width and height
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Reduce the resolution if it's too high to avoid memory issues
    resize_factor = 1  # Change this factor to reduce resolution, e.g., 0.5 to reduce to half
    resized_width = int(original_width * resize_factor)
    resized_height = int(original_height * resize_factor)

    # Set the video writer function for saving the output video
    out_video_name = f"{video_path.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(f"{out_video_name}_test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (resized_width, resized_height))

    frame_count, total_fps = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize frame
            frame = cv2.resize(frame, (resized_width, resized_height))
            
            # preprocess frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = letterbox(image, new_shape=resized_width, auto=False)[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image is image.float()
            start_time = time.time()

            # Inference
            output, _ = model(image)
            # Apply non-max suppression to keypoints
            output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                             kpt_label=True)
            # Transform outputs to keypoints
            output = output_to_keypoint(output)
            img = image[0].permute(1, 2, 0) * 255
            img = img.cpu().numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            icon = cv2.imread("icon3.png")
            # Fall detection
            thre = (resized_height // 2) + 100
            for idx in range(output.shape[0]):
                kpts = output[idx, 7:].T
                plot_skeleton_kpts(img, kpts, 3)
                cx, cy, w, h = output[idx, 2], output[idx, 3], output[idx, 4], output[idx, 5]
                xmin, ymin = int(cx - w / 2), int(cy - h / 2)
                xmax, ymax = int(cx + w / 2), int(cy + h / 2)
                p1 = (int(xmin), int(ymin))
                p2 = (int(xmax), int(ymax))
                cx, cy = int(cx), int(cy)  # Ensure cx and cy are integers
                dx, dy = xmax - xmin, ymax - ymin
                icon = cv2.resize(icon, (50, 50), interpolation=cv2.INTER_LINEAR)
                difference = dy - dx
                ph = Get_coord(kpts, 2)
                fall_detected = ((difference < 0) and (ph > thre)) or (difference < 0)
                
                if fall_detected:
                    draw_border(img, p1, p2, (84, 61, 247), 10, 25, 25)
                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    draw.rounded_rectangle((cx - 10, cy - 10, cx + 60, cy + 60), fill=(84, 61, 247), radius=15)
                    img = np.array(im)
                    if (0 <= cy < resized_height - 50) and (0 <= cx < resized_width - 50):
                        img[cy:cy + 50, cx:cx + 50] = icon

                # Save keypoints and bounding box data to CSV
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for i in range(0, len(kpts), 3):
                        if i + 2 < len(kpts):
                            kpt_x, kpt_y, kpt_conf = kpts[i:i + 3]
                            if kpt_conf > 0.5:  # Confidence threshold
                                writer.writerow(
                                    [frame_count, xmin, ymin, xmax, ymax, kpt_x, kpt_y, kpt_conf, abs(kpt_x - cx),
                                     abs(kpt_y - cy), fall_detected])

            # Show the preprocessed image
            cv2.imshow("Detection", img)
            cv2.waitKey(1)
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            # Write the frame into the output video file
            out.write(img)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))