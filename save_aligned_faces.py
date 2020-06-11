import os
import sys
import cv2
import torch
import copy
import pickle


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
 
    return x1, y1, size_bb


def adjust_boundingbox(face, width, height, size):
    """
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param size: set bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = min(max(int(center_x - size // 2), 0), width - size)
    y1 = min(max(int(center_y - size // 2), 0), height - size)
    
    return [x1, y1, size]


def vanilla_bbox_iou_overlaps(b1, b2):
    """
    Arguments:
        b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-union pair-wise.
    """
    area1 = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
    area2 = (b2[:, 2] - b2[:, 0] + 1) * (b2[:, 3] - b2[:, 1] + 1)
    lt = torch.max(b1[:, None, :2], b2[:, :2])
    rb = torch.min(b1[:, None, 2:4], b2[:, 2:4])
    wh = (rb - lt + 1).clamp(min=0)
    inter_area = wh[:, :, 0] * wh[:, :, 1]
    union_area = area1[:, None] + area2 - inter_area
    return inter_area / torch.clamp(union_area, min=1)


def save_aligned_face(video_path, save_video_path):
    file_paths = os.listdir(video_path)
    frame_paths = []
    for path in file_paths:
        if path.startswith('frame') and path.endswith('.png'):
            frame_paths.append(os.path.join(video_path, path))
            
    info = {}
    
    height, width = None, None
    for retry in range(3):
        success = False
        try:
            img = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
            height, width = img.shape[:2]
            success = True
        except:
            pass
        if success:
            break
    if height is None:
        print('Error! Cannot read from %s'%frame_paths[0])
        return
    info['height'] = height
    info['width'] = width
        
    num_frames = len(frame_paths)
    frame_structs = [None for _ in range(num_frames+1)]
    voter = {}
    for path in frame_paths:
        frame_idx = int(path.split('/')[-1].split('.')[0][5:])
        
        bbox_path = path.replace('DFDC-Kaggle_image', 'DFDC-Kaggle_Retinaface')
        bbox_path = bbox_path[:-3] + 'txt'
        face_dic = []
        try:
            with open(bbox_path, 'r') as f:
                for line in f.readlines():
                    load_dict = line[:-1]
                    face_dic.append(load_dict)
        except:
            print('Missing! No %s'%bbox_path)
            pass

        frame_struct = [None]
        for idx in range(1, len(face_dic)):
            load_dict = face_dic[idx]
            score = float(load_dict.split(' ')[4])
            # skip faces with score less than 0.8
            if score < 0.8:
                continue
                
            face = load_dict.split(' ')[:4]
            for _ in range(4):
                face[_] = int(face[_])
            face[2] = face[0] + face[2]
            face[3] = face[1] + face[3]
            x, y, size = get_boundingbox(
                face, width, height, scale=1.3)
            
            frame_struct.append({'score': score, 'bbox': face, 'crop_bbox': [x, y, size], 
                                 'original_idx': idx, 'source': path})
            
        frame_structs[frame_idx] = frame_struct
        num_faces = len(frame_struct) - 1
        if num_faces not in voter:
            voter[num_faces] = 0
        voter[num_faces] += 1
    
    max_cnt, num_faces = 0, -1
    for num in voter:
        if voter[num] > max_cnt:
            max_cnt = voter[num]
            num_faces = num
    prefix = ''
    if max_cnt < num_frames or num_faces <= 0:
        prefix = 'Caution! '
    print('%s%d / %d of the frames agree on %d face(s) in %s'%(prefix, max_cnt, num_frames, num_faces, video_path))
    sys.stdout.flush()
    
    info['num_frames'] = num_frames
    info['num_faces'] = num_faces
    if num_faces <= 0:
        print('Skipped! No face in %s'%video_path)
        sys.stdout.flush()
        
        mkdir(save_video_path)
        with open(os.path.join(save_video_path, 'info.pkl'), 'wb') as f:
            pickle.dump(info, f)
            
        return
    
    # Connect faces to tubes based on IoU
    active_faces = None
    face_tubes = []
    for frame_idx in range(1, len(frame_structs)):
        frame_struct = frame_structs[frame_idx]
        if len(frame_struct) <= 1:
            continue
            
        cur_faces = [frame_struct[_]['bbox'] for _ in range(1, len(frame_struct))]
        cur_faces = torch.FloatTensor(cur_faces)
        if active_faces is not None:
            ious = vanilla_bbox_iou_overlaps(cur_faces, active_faces)
            max_iou, max_idx = ious.max(dim=1)
            mark = [False for _ in range(len(active_faces))]
        else:
            max_iou, max_idx = None, None
            
        for face_idx in range(1, len(frame_struct)):
            idx = face_idx - 1
            # IoU threshold 0.5 for determining whether is the same person
            if max_iou is None or max_iou[idx] < 0.5:
                if active_faces is None:
                    active_faces = cur_faces[idx].unsqueeze(0)
                else:
                    active_faces = torch.cat([active_faces, cur_faces[idx].unsqueeze(0)], dim=0)
                face_tubes.append([[frame_idx, face_idx]])
            else:
                correspond_idx = max_idx[idx]
                # Each face tube can only add at most one face from a frame
                if mark[correspond_idx]:
                    continue
                mark[correspond_idx] = True
                active_faces[correspond_idx] = cur_faces[idx]
                face_tubes[correspond_idx].append([frame_idx, face_idx])
    
    # Choose num_faces longest face_tubes as chosen faces
    face_tubes.sort(key=lambda tube: len(tube), reverse=True)
    face_tubes = face_tubes[:num_faces]
    
    output, incomplete = '', False
    for idx in range(num_faces):
        output += 'tube%d\t%d / %d'%(idx + 1, len(face_tubes[idx]), num_frames)
        if idx < num_faces - 1:
            output += '\t'
        if len(face_tubes[idx]) < num_frames:
            incomplete = True
    if incomplete:
        print('Incomplete face tube(s) at %s: %s'%(video_path, output))
        sys.stdout.flush()
    
    saved_faces = {}
    for frame_idx in range(1, num_frames + 1):
        saved_faces['frame%d'%frame_idx] = {}
        
    for face_idx in range(1, num_faces + 1):
        idx = face_idx - 1
        # record max crop_bbox size
        tube_idx, max_size = 0, 0
        for frame_idx in range(1, num_frames + 1):
            cur_face = face_tubes[idx][tube_idx]
            next_face = None if tube_idx == len(face_tubes[idx]) - 1 else face_tubes[idx][tube_idx + 1]
            if next_face is not None and abs(cur_face[0] - frame_idx) > abs(next_face[0] - frame_idx):
                tube_idx += 1
                cur_face = next_face
            face = copy.deepcopy(frame_structs[cur_face[0]][cur_face[1]])
            saved_faces['frame%d'%frame_idx]['face%d'%face_idx] = face
            
            if face['crop_bbox'][2] > max_size:
                max_size = face['crop_bbox'][2]
       
        max_size = max_size // 2 * 2
        max_size = min(max_size, height, width)
        for frame_idx in range(1, num_frames + 1):
            adjusted_crop_bbox = adjust_boundingbox(saved_faces['frame%d'%frame_idx]['face%d'%face_idx]['bbox'], 
                                                    width, height, max_size)
            saved_faces['frame%d'%frame_idx]['face%d'%face_idx]['crop_bbox'] = adjusted_crop_bbox
    
    info['face_tubes'] = face_tubes
    info['saved_faces'] = saved_faces
    
    mkdir(save_video_path)
    with open(os.path.join(save_video_path, 'info.pkl'), 'wb') as f:
        pickle.dump(info, f)
    
    last_source = ''
    for frame_idx in range(1, num_frames + 1):
        for face_idx in range(1, num_faces + 1):
            face = saved_faces['frame%d'%frame_idx]['face%d'%face_idx]
            try:
                if face['source'] != last_source:
                    img = cv2.imread(face['source'], cv2.IMREAD_COLOR)
                    last_source = face['source']
                    
                prefix = 'frame%d_face%d'%(frame_idx, face_idx)
                x, y, size = face['crop_bbox']

                face_img = img[y:y + size, x:x + size]
                save_face_path = os.path.join(save_video_path, prefix + '.png')
                cv2.imwrite(save_face_path, face_img)
                # also save a jpg version
                save_face_path = os.path.join(save_video_path, prefix + '.jpg')
                cv2.imwrite(save_face_path, face_img)
            except Exception as e:
                print(e)
                print('Error! Failed to save face image(s) from %s'%path)
                sys.stdout.flush()
            
    print('Success! Faces at %s finished saving to %s'%(video_path, save_video_path))
    sys.stdout.flush()
                

if __name__ == '__main__':
    root_dir = 'DFDC-Kaggle_image/dfdc_train_part_'
    save_root_dir = 'DFDC-Kaggle_Alignedface/dfdc_train_part_'
    video_list, save_video_list = [], []
    for part in range(50):
        part_dir = root_dir + str(part)
        save_part_dir = save_root_dir + str(part)
        vids = os.listdir(part_dir)
        for vid in vids:
            video_list.append(os.path.join(part_dir, vid))
            save_video_list.append(os.path.join(save_part_dir, vid))

    print(len(video_list))
    
    for i in range(len(video_list)):
        save_aligned_face(video_list[i], save_video_list[i])
