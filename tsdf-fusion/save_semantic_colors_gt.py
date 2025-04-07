import numpy as np
import cv2
import glob
import random
import os
import imageio
import os


def generate_random_colors(N):
    colors = []
    for _ in range(N):
        # Generate random RGB color for visualization color code
        color = [random.randint(0, 255) for _ in range(3)] 
        colors.append(color)
    return np.array(colors)

sem_path = '/media/saimouli/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/semantic_class'
sem_save = '/media/saimouli/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/semantic_color'
os.makedirs(sem_save, exist_ok=True)

random_colors = generate_random_colors(225)

# Save the color code for later evaluation phase to fetch index.
np.save(f'{sem_save}/../color_code.npy', random_colors) 

sems = sorted(glob.glob(f"{sem_path}/semantic_class_*.png"))
for sem_path in sems:
    name = os.path.basename(sem_path)
    sem = cv2.imread(sem_path, -1).astype(int)
    new_color = random_colors[sem]
    imageio.imwrite(f"{sem_save}/{name}", new_color.astype(np.uint8))