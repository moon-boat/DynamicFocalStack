import numpy as np
import pandas as pd
import os
import torch
from matplotlib import pyplot as plt 
import matplotlib as mpl
import cv2
import glob
import time

from model import AddNet, MergeStackNet
from utils import Options,  Dataset_Test, Dataset_Test_Real, downsample_3D


class EventData:
    def __init__(self, h=None, w=None):
        # self.data用numpy矩阵表示，形状为4*N，pol为+1或-1
        # 第零列是t，第一列是x（即j），第二列是y（即i），第三列是pol
        self.data = None
        self.h = h
        self.w = w
    
    # txt格式里，pol为1或0
    def read_txt(self, filename):
        print(filename)
        self.data = pd.read_csv(filename, delim_whitespace=True, header=None, names=['t', 'x', 'y', 'pol'],
                                        dtype={'t': np.int64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                        engine='c', skiprows=0, nrows=None).values
        print(self.data.shape)
        self.data[:, 3] = np.where(self.data[:, 3]==1, 1, -1)
        return self.data

    def write_txt(self, filename):
        out = self.data.copy()
        out[:, 3] = np.where(self.data[:, 3]==1, 1, 0)
        with open(filename, "w") as f:
            for e in out:
                f.write("%d %d %d %d", e[0], e[1], e[2], e[3])
        return 0
    

    def read_npy(self, filename):
        self.data = np.load(filename)
        self.data[:, 3] = np.where(self.data[:, 3]==1, 1, -1)
        return self.data
    
    def write_npy(self, filename):
        np.save(filename, self.data)
    
    def get_h_w(self):
        if self.h == None:
            self.h = np.max(self.data[:,2])+1
        if self.w == None:
            self.w = np.max(self.data[:,1])+1
        return self.h, self.w

    def get_t(self):
        t = np.max(self.data[:,0])
        return t
        
        
    def to_stack(self, layers):
        self.get_h_w()
        stack = np.zeros(layers, self.h, self.w)
        sub = np.array_split(self.data, layers)
        for i in range(layers):
            i_col = sub[i][:, 2]
            j_col = sub[i][:, 1]
            p_col = sub[i][:, 3]
            np.add.at(stack[i], (i_col, j_col), p_col)
        return stack
    
    def ev2img_zxy(self, matrix, cl=3):
        norm = mpl.colors.Normalize(vmin=-cl, vmax=cl, clip=True)
        color = mpl.cm.get_cmap("bwr")
        n = norm(matrix)
        c = color(n)
        return np.uint8(c*255)[:,:,0:3]
    
    # startp & endp 在[0,1]之间，表示要画哪一小段
    def to_image(self, cl=3, startp=0, endp=1):
        self.get_h_w()
        evcnt = self.data.shape[0]
        start = int(startp*evcnt)
        end = int(endp*evcnt)
        i_col = np.int32(self.data[start:end, 2])
        j_col = np.int32(self.data[start:end, 1])
        p_col = np.where(self.data[start:end, 3]==1, 1, -1)
        stack = np.zeros((1, self.h, self.w))
        np.add.at(stack[0], (i_col, j_col), p_col)
        img = self.ev2img_zxy(stack[0], cl)
        return img
    
    # 4*h*w, c0表示正event数量，c1表示负event数量，c2表示最新正event时间，c3表示最新负event时间
    def to_4c_timesurface(self):
        self.get_h_w()
        img = np.zeros((4, self.h, self.w))
        mintime = np.min(self.data[:,0])
        maxtime = np.max(self.data[:,0])
        t_col = (self.data[:,0] - mintime) / (maxtime-mintime)
        i_col = np.int32(self.data[:, 2])
        j_col = np.int32(self.data[:, 1])
        p_col = np.int32(self.data[:, 3])
        np.add.at(img[0], (i_col[p_col==1], j_col[p_col==1]), 1)
        np.add.at(img[1], (i_col[p_col<1], j_col[p_col<1]), 1)
        img[2, i_col[p_col==1], j_col[p_col==1]] = t_col[p_col == 1]
        img[3, i_col[p_col<1], j_col[p_col<1]] = t_col[p_col < 1]
        return img
    
    def to_video(self, outpath, frame_cnt=16, clipn=3):
        self.get_h_w()
        video = cv2.VideoWriter(outpath, 0, 20, (self.w, self.h))
        event_cnt = self.data.shape[0]
        epf = int(event_cnt // frame_cnt)
        for i in range(frame_cnt):
            begin = max(0, int(epf*(i-3)))
            end = min(event_cnt, int(epf*(i+3)))
            stack = np.zeros((self.h, self.w))
            i_col = np.int32(self.data[begin:end, 2])
            j_col = np.int32(self.data[begin:end, 1])
            p_col = np.where(self.data[begin:end, 3]==1, 1, -1)
            np.add.at(stack, (i_col, j_col), p_col)
            color = self.ev2img_zxy(stack, clipn)
            video.write(color)
        cv2.destroyAllWindows()
        video.release()
        return 0

def normalize_uint8(arr):
    maxval = np.max(arr)
    minval = np.min(arr)
    return np.uint8((arr - minval)*255.0 / (maxval-minval))

def colorRB(img):
    temp = np.zeros([img.shape[0], img.shape[1], 3]) + 255
    channel_r = temp[:, :, 0]
    channel_g = temp[:, :, 1]
    channel_b = temp[:, :, 2]
    factor = 1
    channel_r[img > 0] = 255 - img[img > 0]*(255)*factor
    channel_g[img > 0] = 255 - img[img > 0]*(255)*factor
    channel_b[img > 0] = 255

    channel_r[img < 0] = 255
    channel_g[img < 0] = 255 + img[img < 0]*(255)*factor
    channel_b[img < 0] = 255 + img[img < 0]*(255)*factor
    temp[:, :, 0] = channel_r
    temp[:, :, 1] = channel_g
    temp[:, :, 2] = channel_b
    return temp

def normalize(matrix):
    minv = np.min(matrix)
    maxv = np.max(matrix)
    matrix[matrix>0] = matrix[matrix>0]/max(maxv, 0.001)
    matrix[matrix<0] = matrix[matrix<0]/min(abs(minv), -0.001)
    return matrix

def single_ev2img(matrix):
    matrix = normalize(matrix)
    matrix = colorRB(matrix)
    return np.uint8(matrix)
    
def ev2img(matrix):
    # input matrix size is [N,H,W]
    # Convert to RGB visualization events
    matrix = np.float32(np.sum(matrix, axis=0))
    matrix = normalize(matrix)
    matrix = colorRB(matrix)
    return np.uint8(matrix)

def ev2img_zxy(matrix, cl=3):
    norm = mpl.colors.Normalize(vmin=-cl, vmax=cl, clip=True)
    color = mpl.cm.get_cmap("bwr")
    n = norm(matrix)
    c = color(n)
    return np.uint8(c*255)[:,:,0:3]

def get_first_index_after(t, events):
    left = 0
    right = events.shape[0]
    min_ans = right - 1
    while left < right:
        mid = (right + left) // 2
        if events[mid][0] <= t:
            left = mid + 1
            continue
        else:
            min_ans = mid
            right = mid - 1
    return min_ans

def stack_events(h, w, events, stacksize):
    event_cnt = events.shape[0]
    cnt_per_frame = event_cnt // stacksize
    stack = np.zeros((stacksize, h, w))
    i_col = events[:,2]
    j_col = events[:,1]
    p_col = np.where(events[:,3]==1, 1, -1)
    for i in range(stacksize):
        np.add.at(stack[i], (i_col[i*cnt_per_frame:(i+1)*cnt_per_frame], j_col[i*cnt_per_frame:(i+1)*cnt_per_frame]), p_col[i*cnt_per_frame:(i+1)*cnt_per_frame])
    return stack


# Reconstruct an image and evaluate the sharpness.
def make_image_and_eval(image, events, evrefocusnet, device, voltmeter, frame = 0):
    h, w, c = image.shape
    STACK_SIZE = 64
    evstack = stack_events(h, w, events, STACK_SIZE)
    # In the training data, all event polarities were flipped because there was a bug in the Voltmeter simulator.
    # So we need to flip all correct data to the wrong polarity to get correct results......
    
    if voltmeter != True:
        evstack = - evstack
    
    '''
    if ((frame % 2) == 0):
        evstack = - evstack
    '''
        
    pd = evrefocusnet(torch.Tensor(image.transpose(2,0,1)).to(device).unsqueeze(0), \
        torch.Tensor(evstack).to(device).unsqueeze(0))
    pd_y = torch.mean(pd, dim=1) # get rid of color channels
    sharpness = torch.var(pd_y)
    pd_np = np.uint8(torch.squeeze(pd.cpu()).numpy()).transpose((1,2,0))
    sharpness_np = torch.squeeze(sharpness.cpu()).numpy()
    return pd_np, sharpness_np


# Do the golden rate search to find the refocus timestamps, and reconstruct the refocused images using EvRefocusNet.
def make_stack_images(imgpath, evpath, stackpath, patch_N, evrefocusnet, device, voltmeter=False, output_patches=False, frame = 0):
    img = cv2.imread(imgpath)
    
    # print(f"imgpath is {imgpath}")
    
    h, w, c = img.shape
    
    events = np.int32(np.load(evpath))
    # print(f"event {frame} exists negative event with number {np.sum(events < 0)}")
    
    events_binned = np.zeros((patch_N, patch_N), dtype=object)
    
    patch_h = ((h // patch_N) // 4 + 1) * 4
    patch_w = ((w // patch_N) // 4 + 1) * 4

    def patch_borders(i, j):
        mi = patch_h*i
        mj = patch_w*j
        Mi = patch_h*(i+1)
        Mj = patch_w*(j+1)
        if Mi > h:
            mi = h - patch_h
            Mi = h
        if Mj > w:
            mj = w - patch_w
            Mj = w
        return mi, Mi, mj, Mj
    # Split one (n, 4) event list into N*N sublists, such that events_binned[i,j] contains all events in patch (i,j).
    for i in range(patch_N):
        for j in range(patch_N):
            mi, Mi, mj, Mj = patch_borders(i, j)
            in_patch = np.logical_and( \
                np.logical_and(events[:,2]>=mi, events[:,2]<Mi), \
                np.logical_and(events[:,1]>=mj, events[:,1]<Mj))
            events_binned[i,j] = events[in_patch]
            events_binned[i,j][:,2] -= mi
            events_binned[i,j][:,1] -= mj
    
    # For each patch, search for in-focus moment with golden-rate-search.
    # This could be done in parallel.
    golden_ratio = (1+5**0.5)/2
    event_cnt_thres = 500
    
    found_times = []

    for i in range(patch_N):
        for j in range(patch_N):
            mi, Mi, mj, Mj = patch_borders(i, j)
            imgpatch = img[mi:Mi, mj:Mj]
            evpatch = events_binned[i,j]
            if evpatch.shape[0] < event_cnt_thres*2:
                # Not enough events to run algorithm. Means that the patch doesn't change and probably doesn't have useful texture.
                # But we still need N*N frames
                found_times.append(0)
                # print(f"evpatch.shape[0] is {evpatch.shape[0]}, event_cnt_thres is {event_cnt_thres}")
                # print(f"event_binned is {events_binned}")
                # exit()
                continue

            left = 0
            right = evpatch.shape[0]
            while right - left > event_cnt_thres:
                t1 = max(left+1, int(right - (right-left)/golden_ratio))
                t2 = min(right-1, int(left + (right-left)/golden_ratio))
                # left < t1 < t2 < right

                img1, val1 = make_image_and_eval(imgpatch, evpatch[0:t1], evrefocusnet, device, voltmeter, frame)
                img2, val2 = make_image_and_eval(imgpatch, evpatch[0:t2], evrefocusnet, device, voltmeter, frame)

                if (val1 < val2):
                    left = t1
                else:
                    right = t2
            found_time = (left + right) // 2
            t = evpatch[found_time,0]
            ti_global = get_first_index_after(t, events)
            found_times.append(ti_global)

            if output_patches:
                # 整张图
                my_img, my_val = make_image_and_eval(img, events[0:ti_global], evrefocusnet, device, voltmeter)
                border_color = np.array([255, 255, 255])
                bwidth = 3
                # 描四条边
                my_img[mi:Mi, mj:mj+bwidth] = border_color
                my_img[mi:Mi, Mj-bwidth:Mj] = border_color
                my_img[mi:mi+bwidth, mj:Mj] = border_color
                my_img[Mi-bwidth:Mi, mj:Mj] = border_color
                cv2.imwrite(stackpath+"patch_img_%02d_%02d.png"%(i,j), my_img)
    
    init_time = events[0][0]
    max_time = np.max(events[:, 0])
    
    
    print(f"init time is {init_time}, max time is {max_time}")
    time_bin_cnt = 64
    print(f"the frame number is {frame}")

    '''
    selected_times = []
    
    time_interval = (max_time - init_time) // time_bin_cnt + 1
    for i in range(64):
        t = init_time + i * time_interval
        index = get_first_index_after(t, events)
        selected_times.append(index)
    '''
    
    selected_times = sorted(found_times)
    
    event_stack = np.zeros((64, h, w))
    
    print(f"select_times are {selected_times}")
    
    frame_cnt = 0
    output_stack = np.zeros((len(selected_times), h, w, c))
    base_path = "/home/xsh/DFS/stacks/event_0/"
    i = 0
    for t in selected_times:
        big_img, big_val = make_image_and_eval(img, events[0:t], evrefocusnet, device, voltmeter, frame)

        output_stack[frame_cnt] = big_img
        cv2.imwrite(base_path + f"{i}.jpg", big_img)
        i += 1

        init_time = events[0][0]
        max_time = np.max(events[:, 0])
        time_bin_cnt = 64

        time_interval = (max_time - init_time) // time_bin_cnt + 1
        cur_time = events[t,0]
        s_min = get_first_index_after(cur_time - time_interval, events)
        s_max = get_first_index_after(cur_time + time_interval, events)
        pols = np.where(events[s_min:s_max, 3] == 0, -1, 1)
        np.add.at(event_stack[frame_cnt], (events[s_min:s_max, 2], events[s_min:s_max, 1]), pols)

        frame_cnt += 1

   
    return output_stack, event_stack

# Merge with gradients. Used for ablation studies.
def merge_stack_gradient(stack):
    # Stack: (stacksize, h, w, c)
    ss, h, w, c = stack.shape

    weights = np.zeros((ss, h, w))
    for x in range(ss):
        dx = cv2.Sobel(stack[x], ddepth=-1, dx=1, dy=0).sum(axis=2)
        dy = cv2.Sobel(stack[x], ddepth=-1, dx=0, dy=1).sum(axis=2)
        gradient = dx*dx + dy*dy
        weights[x] = cv2.GaussianBlur(gradient, (41,41), sigmaX=7, sigmaY=7)
    
    weight_sum = weights.sum(axis=0).reshape((1, h, w))
    weights_norm = np.where(weight_sum < 1e-4, 0, weights / weight_sum).reshape((ss, h, w, 1))

    val = (stack * weights_norm).sum(axis=0)

    weight_indexes = np.indices(stack.shape)[0]
    weight_avg_index = (weights_norm*weight_indexes).sum(axis=0)
    weight_viz = cv2.applyColorMap(normalize_uint8(weight_avg_index.sum(axis=2)), 19)
    
    return val, weight_viz

# Use EvMergeNet to merge stack.
def merge_stack_with_model(imgstack, evstack, evmergenet, device):
    ss, h, w, c = imgstack.shape
    
    # torch.cuda.empty_cache()
    
    pd, weights, _ = evmergenet(torch.Tensor(imgstack.transpose(0, 3, 1, 2)).to(device).unsqueeze(0), \
        torch.Tensor(evstack).to(device).unsqueeze(0))
    pd_np = np.uint8(torch.squeeze(pd.cpu()).numpy()).transpose((1,2,0))
    weights_np = torch.squeeze(weights.cpu())

    weight_sum = weights_np.sum(axis=0).reshape((1, h, w))
    weights_norm = np.where(weight_sum < 1e-4, 0, weights_np / weight_sum).reshape((ss, h, w, 1))
    weight_indexes = np.indices(imgstack.shape)[0]
    weight_avg_index = (weights_norm*weight_indexes).sum(axis=0)
    weight_viz = cv2.applyColorMap(normalize_uint8(weight_avg_index.sum(axis=2)), 19)
    
    return pd_np, weight_viz




def test_all_data_in_path(base_path, model_add, model_merge, device):
    os.makedirs(base_path + "all_in_focus_result/", exist_ok=True)
    os.makedirs(base_path + "weights/", exist_ok=True)
    patch_N = 8

    datanames = [x.split("/")[-1].split("\\")[-1].split(".")[0] for x in sorted(glob.glob(base_path+"input/events_more/*.npy"))]
    for dataname in datanames:
        print(dataname)
        
        import re

        match = re.search(r'event_(\d+)', dataname)

        if match:
            number = int(match.group(1))
            print("Extracted number:", number)
        else:
            print("No match found")

        
        impath = base_path+"input/frames_v1/0000_"+ f"{number:06}" +".jpg"
        evpath = base_path+"input/events_mm_g_v1/"+ dataname +".npy"
        start_time = time.time()

        OUTPUT_PATCHES = False
        
        stackpath = base_path + "stacks/" + dataname + "/"
        os.makedirs(stackpath, exist_ok=True)
        
        if OUTPUT_PATCHES:
            os.makedirs(stackpath, exist_ok=True)
            
        imgstack, evstack = make_stack_images(impath, evpath, stackpath, patch_N, model_add, device, voltmeter=False, output_patches=OUTPUT_PATCHES, frame = number)

        stackpath = "/home/xsh/DFS/vis_events/"
        os.makedirs(stackpath, exist_ok=True)
        
        vis_path = stackpath + f"{number}/"
        os.makedirs(vis_path, exist_ok=True)
        for k in range(64):
            event_vis = single_ev2img(evstack[k])
            # print(f"event {i} in stack {k} has negative event with number {np.sum(evstack[k]<0)}")
            path = vis_path + f"{k}.jpg"
            cv2.imwrite(path, event_vis)

        mergepath = base_path + "all_in_focus_result/" + dataname + ".jpg"
        
        img_path = "/home/xsh/DFS/img_stacks/"
        os.makedirs(img_path, exist_ok=True)
        img_path_rela = img_path + dataname + "/"
        os.makedirs(img_path_rela, exist_ok=True)
        for j,img in enumerate(imgstack):
            stack_img_path = img_path_rela + f"{j}.jpg"
            cv2.imwrite(stack_img_path, img)
            
        # torch.cuda.empty_cache()
        
        with torch.no_grad():
            aif, weights = merge_stack_with_model(imgstack, evstack, model_merge, device)
        cv2.imwrite(mergepath, aif)
        
        weightpath = base_path + "weights/" + dataname + ".jpg"
        cv2.imwrite(weightpath, weights)

        end_time = time.time()
        print("Used seconds: %d" % int(end_time-start_time))
        
if __name__ == "__main__":
    device = torch.device("cuda")
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    
    StackSize = 64
    model_add = AddNet(StackSize)
    model_merge = MergeStackNet(StackSize)
    model_add = torch.nn.DataParallel(model_add.cuda(), device_ids=[device_ids[0]], output_device=device_ids[0])
    model_merge = torch.nn.DataParallel(model_merge.cuda(), device_ids=[device_ids[0]], output_device=device_ids[0])
    model_add.load_state_dict(torch.load("/home/xsh/DFS/pretrained/model_best copy.pth", map_location='cpu'), strict=False)
    model_merge.load_state_dict(torch.load("/home/xsh/DFS/pretrained/mergenet_best.pth", map_location='cpu'), strict=False)
    model_add.eval()
    model_merge.eval()
    
    
    base_path = "/home/xsh/DFS/"
    N = 64
    with torch.no_grad():
        test_all_data_in_path(base_path, model_add, model_merge, device)