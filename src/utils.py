import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from itertools import permutations, combinations
import cv2

import matplotlib.cm as cm
from PIL import Image
import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Low_pass(matrix):
    dtf = cv2.dft(matrix, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dtf)
    h, w = matrix.shape
    h2, w2 = h // 2, w // 2  # 中心点
    dft_shift[0:h2 - 6, :] = 0
    dft_shift[h2 + 6:, :] = 0
    dft_shift[:, :w2 - 6] = 0
    dft_shift[:, w2 + 6:] = 0
    ifft_shift2 = np.fft.ifftshift(dft_shift)
    result = cv2.idft(ifft_shift2)
    return result[:,:,0]

def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def thresh_heatmap_get_pred_point(heatmap_list,threshold):
    num_position = len(heatmap_list)
    print('total:{}'.format(num_position))

    CLS_coords_list = []
    for round_index in range(num_position):
        CLS_coords_list.append([])
        heatmap = (heatmap_list[round_index] * 255).reshape(60, 60)
        vis_pred_test = np.maximum(heatmap, 0)
        vis_pred_test = MaxMinNormalization(vis_pred_test)
        matrix = np.array(vis_pred_test > threshold).astype(float)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(matrix))
        cadidate_value_list = []
        coords_dict = {}

        for connected_index in range(1, len(centroids)):
            [pixel_col, pixel_row] = [stats[connected_index][0] + stats[connected_index][2] / 2,
                                      stats[connected_index][1] + stats[connected_index][3] / 2]
            center_value = heatmap[int(stats[connected_index][1] + stats[connected_index][3] / 2),
                                   int(stats[connected_index][0] + stats[connected_index][2] / 2)]
            center_300x300_x = -147.5 + pixel_col * 5
            center_300x300_y = 147.5 - pixel_row * 5
            coords_dict[str(center_value)] = [center_300x300_x, center_300x300_y]
            cadidate_value_list.append(center_value)
        cadidate_value_list.sort(reverse=True)  # large to small

        for candidata_index in range(len(cadidate_value_list)):
            value = cadidate_value_list[candidata_index]
            coords = coords_dict[str(value)]
            CLS_coords_list[round_index].append(coords)

    return np.array(CLS_coords_list, dtype=object)



def predict(model, test_set, numGrids=60):
    """
    Function to predict heatmap based on dRSS and prior information.

    Args:
        dRSS: Input dRSS data as a 40x40 array.
        prior: Prior information as a 60x60 array.

    Returns:
        pred_heatmap: Predicted heatmap as a 60x60 array.
    """
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=torch.cuda.is_available()
    )

    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch, (dRSS, _, prior) in enumerate(test_loader):
            dRSS = dRSS.to(device)  # [b, 1, 40, 40]
            prior = prior.to(device)  # [b, 1, 60, 60]
            pre_pred, pred = model(dRSS.float(), prior)
            pred = pred.cpu().numpy().reshape(numGrids, numGrids)
            pred_list.append(pred)
        pred_heatmap = np.array(pred_list)
    return pred_heatmap

def get_pred_coords(input_heatmap, Low_pass_hp=True, threshold=0.45):
    """
    Function to extract coordinates from the predicted heatmap.

    Args:
        input_heatmap: Predicted heatmap as a 2D numpy array.

    Returns:
        pred_coords: List of coordinates [(x1, y1), (x2, y2), ...].
    """
    if Low_pass_hp:
        input_heatmap = [Low_pass(input_heatmap[i].reshape(60,60)) for i in range(len(input_heatmap))]

    pred_coords = thresh_heatmap_get_pred_point(input_heatmap, threshold)

    return pred_coords

def plot_result(result_list, GT_coords_list, pred_coords, visual_index):
    """
    Function to visualizing result by given our predicted result and ground truth.
    """
    GT_coord = GT_coords_list[visual_index]
    ours_pred_coord = pred_coords[visual_index]

    probability_map = np.array(result_list[visual_index])  # 60x60
    cmap = cm.get_cmap('Blues')
    probability_map = (cmap(probability_map) * 255).astype(np.uint8)
    probability_map = Image.fromarray(probability_map, mode='RGBA').resize((300, 300))

    fig, axes = plt.subplots(1, 1, figsize=(15, 15))

    for GT_index in range(len(GT_coord)):
        circle_x = GT_coord[GT_index][0]
        circle_y = GT_coord[GT_index][1]
        #     if test == 'GT':
        circle_x = circle_x
        circle_y = circle_y
        plt.subplot(1, 1, 1)
        circle = Circle((circle_x, circle_y), radius=13, color='black', fill=False, linewidth=5)
        plt.scatter(circle_x, circle_y, s=50, c='black', marker='o', alpha=1, linewidths=0.5)
        axes.add_artist(circle)

    for pred_coord_index in range(len(ours_pred_coord)):
        circle_x = ours_pred_coord[pred_coord_index][0]
        circle_y = ours_pred_coord[pred_coord_index][1]
        circle_x = circle_x - 2.5
        circle_y = circle_y + 2.5
        circle = Circle((circle_x, circle_y), radius=13, color='red', linestyle='--', fill=True, alpha=0.35,
                        linewidth=5)
        plt.scatter(circle_x, circle_y, s=40, c='red', marker='o', alpha=1, linewidths=4)
        axes.add_artist(circle)

    plt.xlim([-150, 150])
    plt.ylim([-150, 150])
    plt.xticks(np.linspace(-150, 150, 7), fontsize=35)
    plt.yticks(np.linspace(-150, 150, 7), fontsize=35)
    plt.set_cmap('Blues')  # 'viridis'

    plt.imshow(probability_map, extent=[-150, 150, -150, 150], aspect='equal', alpha=0.7)
    plt.title(f'sample: {visual_index}', fontsize=35)
    plt.xlabel('X (cm)', fontsize=35)
    plt.ylabel('Y (cm)', fontsize=35)
    plt.show()
