import yaml
import sys
import os
sys.path.append( os.path.dirname(os.path.dirname(os.path.realpath(__file__))) )
from language.autoencoder.model import AutoencoderLight, EncoderDecoderOnline
from language.supervisedNet import LangSupervisedNet
import os
import torch
import torchvision
import open_clip
import numpy as np
import open3d as o3d
import copy
from sklearn.neighbors import NearestNeighbors
from emd import earth_mover_distance


class OpenCLIPNetwork:
    def __init__(self, device):
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "convnext_large_d_320"
        self.clip_model_pretrained = 'laion2b_s29b_b131k_ft_soup'
        self.clip_n_dims = 768
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)

        self.negatives = ("object", "things", "stuff", "texture")
        self.positives = (" ",)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    def encode_text(self, text_list, device):
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed: 32768x512
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
    
    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    
    def set_semantics(self, text_list):
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.semantic_labels]).to("cuda")
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds /= self.semantic_embeds.norm(dim=-1, keepdim=True)
    
    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        # embed: 3xhxwx512
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred
    
    def get_semantic_map_pc(self, sem_map: torch.Tensor) -> torch.Tensor:
        # embed: 3xhxwx512
        N, C = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = self.semantic_embeds
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(N)
        output = torch.mm(sem_map, p.T)
        softmax = torch.softmax(10 * output, dim=-1)
        print("AA", softmax.shape)
        sem_pred = torch.argmax(softmax, dim=-1)
        sem_pred[sem_pred >= pos_num] = -1
        return sem_pred

    # for each level (default, s, m, l) and for each phrase, compute the relevancy matrix
    def get_max_across(self, sem_map):
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1)

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j)
                pos_prob = probs[..., 0:1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)
        
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map

def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()


def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))

def meshwrite_color(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  colors = colors.astype(np.uint8)
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def mesh_parser(txt_file):
    normals = []
    faces = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        ele = line.split(' ')
        try:
            ele0 = float(ele[0])
        except:
            continue
        if len(ele) != 4:
            normal = np.array([float(ele[3]), float(ele[4]), float(ele[5])])
            normals.append(normal)
        else:
            face = np.array([float(ele[1]), float(ele[2]), float(ele[3])])
            faces.append(face)
    return np.asarray(normals), np.asarray(faces)

def pc15_parser(txt_file):
    pts = []
    feats = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        ele = line.split(' ')
        try:
            ele0 = float(ele[0])
        except:
            continue
        pt = np.array([float(ele[0]), float(ele[1]), float(ele[2])])

        feat = np.array([float(ele[3]), float(ele[4]), float(ele[5]), float(ele[6]), float(ele[7]), float(ele[8]), \
                         float(ele[9]), float(ele[10]), float(ele[11]), float(ele[12]), float(ele[13]), float(ele[14]), float(ele[15]),\
                         float(ele[16]), float(ele[17])])
        pts.append(pt)
        feats.append(feat)
    return np.asarray(pts), np.asarray(feats)



### Various Paths
# Replica_v2 for config
path = '/media/saimouli/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/render_config.yaml'
# Your langsplat/ langslam result path to load reconstructed TSDF
load_path = '/media/saimouli/Data6T/Replica/omni_data_result/room_0_small/2025-03-24-06-23-28/psnr/before_opt'
# where to save point cloud/ mesh results
pc_save_path = f'{load_path}/3d_mesh'
# auto-encoder path
ae_ckpt_path = "/home/saimouli/Desktop/Bosch/training_weights_general/ae_149_he.ckpt"
online_ckpt = torch.load(f'{load_path}/online_15_room0.pth')
# color matrix for groundtruth class
color_mat = np.load('/media/saimouli/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/color_code.npy')
# where is the reconstructed groundtruth point cloud
gt_pcd = o3d.io.read_point_cloud("/media/saimouli/Data6T/Replica/omni_data_result/room_0_small/2025-03-24-06-23-28/psnr/before_opt/GT_semantic_pc.ply")

os.makedirs(pc_save_path, exist_ok=True)
normals, faces = mesh_parser(f'{load_path}/semantic_mesh_color.ply')

with open(path, "rb") as file:
    config = yaml.unsafe_load(file)

# positive prompt list in the environment
objects = config['objects']
name_set = set()
for obj in objects:
    if obj['class_id'] != -1:
        name_set.add(obj['class_name'])

positive_list = list(name_set)
sorted(positive_list)

# all classes in the Replica
class_list = ['background']
for c in config['classes']:
    class_list.append(c['name'])

query = ['floor', 'wall', 'window', 'sofa', 'table', 'cushion', 'lamp', 'rug', 'book', 'blinds']
# TOP 10 labels in Replica_v2
# room0: ['floor', 'wall', 'window', 'sofa', 'table', 'cushion', 'lamp', 'rug', 'book', 'blinds']
# room1: ['wall', 'window', 'blinds', 'floor', 'blanket', 'ceiling', 'lamp', 'comforter', 'bed', 'nightstand']
# room2: ['wall', 'chair', 'floor', 'plate', 'vase', 'window', 'indoor-plant', 'table', 'blinds']
# office0: ['wall', 'rug', 'table', 'blinds', 'sofa', 'tv-screen', 'floor', 'chair', 'door']
# office1: ['wall', 'floor', 'pillow', 'blanket', 'blinds', 'desk', 'chair', 'monitor', 'pillar', 'table']
# office2: ['wall', 'floor', 'table', 'sofa', 'panel', 'cushion', 'chair', 'bottle', 'tissue-paper', 'tv-screen']
# office3: ['floor', 'table', 'wall', 'window', 'sofa', 'tablet', 'chair', 'cushion', 'ceiling', 'door']
# office4: ['wall', 'floor', 'chair', 'ceiling', 'panel', 'bench', 'window', 'lamp', 'tv-screen', 'door']

# instantiate autoencoder and openclip
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = OpenCLIPNetwork(device)
clip_model.set_semantics(positive_list)
encoder_hidden_dims = [512, 256, 128, 64, 32]
decoder_hidden_dims = [192, 256, 384, 512, 768]
auto_model = AutoencoderLight(encoder_hidden_dims, decoder_hidden_dims, 768, is_MLP=True).to("cuda")
auto_model = auto_model.load_from_checkpoint(ae_ckpt_path, 
                                  encoder_hidden_dims=encoder_hidden_dims, 
                                  decoder_hidden_dims=decoder_hidden_dims,
                                  is_MLP=True)
auto_model.to("cuda")
auto_model.eval()
online_auto = EncoderDecoderOnline().to("cuda").eval()
online_auto.load_state_dict(online_ckpt)

# load the reconstructed points with features
points, sem_feat = pc15_parser(f"{load_path}/semantic_pc.ply")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
num_points = sem_feat.shape[0]
sem_feat = torch.from_numpy(sem_feat).float().cuda()
online_feat = online_auto.decode(sem_feat).view(num_points, -1)
auto_feat = auto_model.decode(online_feat)
fetch = clip_model.get_semantic_map_pc(auto_feat).cpu().numpy()

# load in GT color_code
gt_points = np.array(gt_pcd.points)
gt_colors = np.array(gt_pcd.colors)
gt_colors_int = (255 * gt_colors).astype(int)
cds = []
emds = []
for i in range(len(query)):
    pred_positive_idx = positive_list.index(query[i])
    gt_positive_idx = class_list.index(query[i])
    K1 = (fetch == pred_positive_idx)
    queried_points = points[K1]
    queried_points = queried_points[::8]
    K2 = (gt_colors_int[:,0] == color_mat[gt_positive_idx, 0]) & \
    (gt_colors_int[:,1] == color_mat[gt_positive_idx, 1]) & \
        (gt_colors_int[:,2]==color_mat[gt_positive_idx, 2])
    gt_queried_points = gt_points[K2]
    gt_queried_points = gt_queried_points[::8]

    if queried_points.shape[0] == 0 or gt_queried_points.shape[0] == 0:
        print(f"Processing class {query[i]}: No query point for pred shape {queried_points.shape[0]} and GT shape {gt_queried_points.shape[0]}")
        continue

    cd = chamfer_distance(queried_points, gt_queried_points)
    cds.append(cd.item())
    with torch.no_grad():
        pt_queried_points = torch.from_numpy(queried_points.astype(np.float32)).cuda()
        pt_gt_queried_points = torch.from_numpy(gt_queried_points.astype(np.float32)).cuda()
        emdd = earth_mover_distance(pt_queried_points, pt_gt_queried_points, transpose=False)
        emds.append(emdd.item())
    print(f"Processing class {query[i]}: CD {cd}, EMD {emdd.item()}")
    del pt_gt_queried_points, pt_queried_points, emdd
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
print(f"average: cd: {sum(cds)/len(cds)}, emd: {sum(emds)/len(emds)}")

true_colors = np.array([200, 100, 0])
false_colors = np.array([110, 110, 110])

for i in range(len(positive_list)):
    clrs = np.where((fetch==i)[..., None], true_colors[None, ...], false_colors[None, ...])
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.colors = o3d.utility.Vector3dVector(clrs)
    xyzrgb  = np.concatenate([pcd_copy.points, pcd_copy.colors], axis=1)
    meshwrite_color(f"{pc_save_path}/{positive_list[i]}_mesh.ply", np.array(pcd_copy.points), faces, normals, np.array(pcd_copy.colors))