import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
import random
import argparse
from model import Detector
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import numpy as np
import pandas as pd
import pickle


def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False, phase='test', resolution_scale=1):
    # Ensure either landmark or bbox is provided
    assert landmark is not None or bbox is not None

    H, W = img.shape[:2]

    # Apply resolution scale to bbox or landmark
    if resolution_scale != 1:
        if bbox is not None:
            # Scale the bounding box coordinates
            bbox = [(int(pt[0] * resolution_scale), int(pt[1] * resolution_scale)) for pt in bbox]
        if landmark is not None:
            # Scale the landmark coordinates
            landmark = landmark * resolution_scale

    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4
        w1_margin = w / 4
        h0_margin = h / 4
        h1_margin = h / 4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 8
        w1_margin = w / 8
        h0_margin = h / 2
        h1_margin = h / 5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand() * 0.6 + 0.2)
        w1_margin *= (np.random.rand() * 0.6 + 0.2)
        h0_margin *= (np.random.rand() * 0.6 + 0.2)
        h1_margin *= (np.random.rand() * 0.6 + 0.2)
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if only_img:
        return img_cropped
    else:
        return img_cropped, None, None, None

def process_real_video(video_path, face_dir,original_vid_frames_dir,image_size=(380, 380)):


    face_list = []
    idx_list = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    video_name = os.path.basename(video_path)
    print(video_name) #000.mp4
    video_base_name = video_name
  
   

    #get num of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_frames) #304


    frame_shape = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_number_str = f"{frame_idx:05d}"

        if frame_shape is None:
            frame_shape = frame.shape



        json_path = os.path.join(face_dir, video_base_name.split(".")[0], f"{video_base_name}_{frame_number_str}.png.json")
        # print(json_path) #/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/face/003/003.mp4_00001.png.json
        
        # /ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/face/003/003.mp4_00304.png.json_00304.png
        #   /ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/face/003/003.mp4_00001.png.json
        # 003.mp4/003.mp4_00001.png.json_00001.png

      

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                bbox_data = json.load(f)
                bbox = bbox_data['bbox']
                x0, y0, x1, y1 = bbox
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(frame.shape[1], x1)
                y1 = min(frame.shape[0], y1)

                bbox_array = np.array([[x0, y0], [x1, y1]])
                face_crop = crop_face(frame, bbox=bbox_array, crop_by_bbox=True, only_img=True, phase='test',resolution_scale=1.0)
                face_crop = cv2.resize(face_crop, image_size)
                face_crop = face_crop.transpose((2, 0, 1))  # Convert to (C, H, W)
                face_list.append(face_crop)
                idx_list.append(frame_idx)
        else:
            # Skip frames without bounding boxes
            # raise ValueError(f"Bounding box not found for {json_path}_{frame_number_str}.png")
            print(f"Bounding box not found for {json_path}_{frame_number_str}.png")
            continue
    cap.release()
    
    return face_list, idx_list, frame_shape,num_frames

def save_results(results, output_pkl='output.pkl'):
        """
        Function to save the results to disk.
        """
        #crate df with append mode
        df = pd.DataFrame(results)
        # print("computing metrics")
        # print("df", df)
        # compute_metrics(df)
        # print("saving df")

    
        # df.to_csv(output_pkl, index=False, mode='a', header=not file_exists)
        # print(f"Data saved to {output_pkl} (appended)")

        # output_pickle = 'output.pkl'

        # Check if the file exists
        file_exists = os.path.exists(output_pkl)

        if file_exists:
            # Load the existing data
            with open(output_pkl, 'rb') as f:
                existing_data =pd.read_pickle(f)
            
            # join the existing df with the new df
            new_data = pd.concat([existing_data, df], ignore_index=True)
            new_data.to_pickle(output_pkl)

        else:
            new_data = df
            new_data.to_pickle(output_pkl)

        # Save the updated data

        print(f"Data saved to {output_pkl}")


def predict(video_path, face_dir,original_vid_frames_dir,debug_dir,model,device, image_size=(380, 380),batch_size=128 * 2):
    
    face_list,idx_list,size,frame_count= process_real_video(video_path,face_dir,original_vid_frames_dir, image_size=(380, 380))
    print(f"Processing {video_path} with {len(face_list)} frames")

    #save to debug
    for i, face in enumerate([face_list[0]]):
        face = face.transpose(1, 2, 0)
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        # print("face", face.shape)
        cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(video_path)}_{i:05d}.png"), face)


    predictions = []
    idxs = []
    with torch.no_grad():
        for i in range(0, len(face_list), batch_size):
            batch_faces = face_list[i:i + batch_size]
            batch_idxs = idx_list[i:i + batch_size]

            batch_faces = torch.tensor(batch_faces).to(device).float() / 255.0
            # batch_faces shape: (N, C, H, W)

            # # Normalize using ImageNet mean and std
            # mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
            # std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
            # batch_faces = (batch_faces - mean) / std

            # print(batch_faces.shape)

            preds = model(batch_faces)
            preds = preds.softmax(1)[:, 1]
            # print(preds)
            predictions.extend(preds.cpu().numpy())
            idxs.extend(batch_idxs)



    # Aggregate predictions
    from collections import defaultdict
    pred_dict = defaultdict(list)
    # print(f"got {len(predictions)} predictions")

    for idx, pred in zip(idxs, predictions):
        pred_dict[idx].append(pred)
    # print(f"pred_dict {pred_dict}")
    pred_res = []
    for idx in sorted(pred_dict.keys()):
        pred_res.append(max(pred_dict[idx]))

    pred_mean = np.mean(pred_res)

    print(f'fakeness: {pred_mean:.4f}')

    return {'video': video_path, 'preds': np.array(pred_res), 'mean': pred_mean, 'size':size,"frame_count": frame_count}

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(args.weight_name, map_location=device)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()


    num_workers = args.num_workers
    worker_id = args.worker_id

    print("worker_id", worker_id)
    print("num_workers", num_workers)


    original_vid_frames_dir = "/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/FF_original_raw/original_sequences/youtube/raw/videos/"

    all_test_videos = os.listdir(original_vid_frames_dir)
    #sort the list
    all_test_videos.sort()

    all_test_videos = [os.path.join(original_vid_frames_dir, v) for v in all_test_videos]
    print("got", len(all_test_videos), "videos")

    

    # Distribute the videos across workers
    def distribute_videos(videos, num_workers, worker_id):
        # Ensure the worker_id is within the range
        if worker_id >= num_workers:
            raise ValueError("worker_id must be less than num_workers")
        
        # Distribute videos by slicing the list
        subset_size = len(videos) // num_workers
        remainder = len(videos) % num_workers
        
        # Calculate start and end indices for this worker's subset
        start_idx = worker_id * subset_size + min(worker_id, remainder)
        end_idx = start_idx + subset_size + (1 if worker_id < remainder else 0)
        
        return videos[start_idx:end_idx]

    # Get the subset of videos for this worker
    test_videos = distribute_videos(all_test_videos, num_workers, worker_id)
    
    print("Number of videos for this worker: ", len(test_videos))


    #now distribute againts the gpus
    num_gpus = args.num_gpus
    gpu_id = args.gpu_id

    print("gpu_id", gpu_id)
    print("num_gpus", num_gpus)

    # Distribute the videos across workers
    def distribute_gpus(videos, num_gpus, gpu_id):
        # Ensure the worker_id is within the range
        if gpu_id >= num_gpus:
            raise ValueError("gpu_id must be less than num_gpus")
        
        # Distribute videos by slicing the list
        subset_size = len(videos) // num_gpus
        remainder = len(videos) % num_gpus
        
        # Calculate start and end indices for this worker's subset
        start_idx = gpu_id * subset_size + min(gpu_id, remainder)
        end_idx = start_idx + subset_size + (1 if gpu_id < remainder else 0)
        
        return videos[start_idx:end_idx]

    # Get the subset of videos for this worker

    test_videos = distribute_gpus(test_videos, num_gpus, gpu_id)

    print(f"Number of videos for this worker {worker_id} and gpu {gpu_id}: ", len(test_videos))


    n_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", n_gpus)

    #get gpu names
    gpu_name = torch.cuda.get_device_name()
    print("GPU name: ", gpu_name)

    
    face_dir = "/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/face/"

    debug_dir = "/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/debug_real/"
    #check and make dir
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)


    output_name = f"pred_worker_{worker_id}_{num_workers}_gpu_{gpu_id}_{num_gpus}.pkl"
    output_txt = os.path.join(args.output_dir, output_name)

    #check if the output file already exists if so add a timestamp
    if os.path.exists(output_txt):
        print("output file already exists RESUMING")

        #load df
        df = pd.read_pickle(output_txt)

        #get the videos that have already been processed
        processed_videos = df["video"].values

        #remove the processed videos from the list
        #remove remove the os.path.join(args.root_path, vid) from the list
        #remove root_path from the path

        print("processed_videos", processed_videos[:5])
        print("processed_videos", len(processed_videos))
        print("test_videos", len(test_videos))
        print("test_videos", test_videos[:5])

        #remove the processed videos from the list
        prev_len = len(test_videos)
        test_videos = [vid for vid in test_videos if vid not in processed_videos]

        print("non processed videos", len(test_videos), "from", prev_len)




    results = []
    for i, video in enumerate(tqdm(test_videos)):
        result = predict(video, face_dir,original_vid_frames_dir,debug_dir, model,device, image_size=(380, 380))
        print(f"vid {result['video']} fakeness: {result['mean']:.4f} size: {result['size']}")
        results.append(result)
        

        if len(results) % args.save_every == 0:
            # # Save results to disk
            # # print("Saving results to disk")
            # # print("results", results)
            save_results(results, output_pkl=output_txt)
            results = []


    # Save any remaining results
    if len(results) > 0:
        # print("Saving results to disk")
        save_results(results)
        results = []

    print("All videos processed.")


        

if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', dest='videolist', type=str, required=True, help='Path to the input video')
    parser.add_argument('-w', dest='weight_name', type=str, help='Path to the model weights',default='/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/SelfBlendedImages/src/weights/FFraw.tar')
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--worker_id", type=int, default=0)

    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--save_every", type=int, default=10)

    #output_dir
    parser.add_argument("--output_dir", type=str, default='/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/SelfBlendedImages/preds_original_vids/')

    args = parser.parse_args()

    #chekc if the output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    main(args)