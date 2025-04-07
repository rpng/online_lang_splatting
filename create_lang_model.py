from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from language.load_lang_model import load_lang_model
from language.autoencoder.model import Autoencoder, AutoencoderLangsplat
from language.sed import add_sed_config
import numpy as np
import cv2
import argparse
import torch
import os
import open_clip
import tqdm
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from argparse import ArgumentParser

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    #get location of current file directory
    currentdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "language/configs/convnextL_768.yaml")
    print("currentdir: ", currentdir)
    parser.add_argument(
        "--config-file",
        default=currentdir,
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    return parser

def get_lang_feat(inputs, model):
    with torch.no_grad():
        _, dense_clip_viz = model(inputs)
        #clip_viz_dense = dense_clip_viz
    return dense_clip_viz

def get_user_embed():
    name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    tokenizer = open_clip.get_tokenizer(name)
    texts = tokenizer(["door"]).cuda()
    #print("texts shape: ", texts.shape)
    clip_model, _, _ = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrain,
            device="cuda",)
    text_embs = clip_model.encode_text(texts)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)
    print("text_embs shape: ", text_embs.shape)
    return text_embs

def save_npy_files(aug, model):
    input_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test/results"
    output_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test"

    ckpt_path = f'auto_ckpt/replica_room0/best_ckpt.pth'
    checkpoint = torch.load(ckpt_path)

    encoder_hidden_dims = [384, 192, 96, 48, 24, 12, 3]
    decoder_hidden_dims = [12, 24, 48, 96, 192, 384, 768]
    auto_model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    auto_model.load_state_dict(checkpoint)
    auto_model.eval()

    # Ensure the output directory exists
    
    lang_feat_dir = os.path.join(output_dir, "language_feat")
    lang_feat_dim3 = os.path.join(output_dir, "language_feat_dim3")
    os.makedirs(lang_feat_dir, exist_ok=True)
    os.makedirs(lang_feat_dim3, exist_ok=True)

    img_save_dir = os.path.join(lang_feat_dim3, "images")
    os.makedirs(img_save_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") and filename.startswith("frame"):
            image_path = os.path.join(input_dir, filename)
            original_image = read_image(image_path, format="BGR")
            height, width = original_image.shape[:2]
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to("cuda")
            inputs = [{"image": image, "height": height, "width": width}]
            clip_viz_dense = get_lang_feat(inputs, model)
            clip_viz_dense_dim3 = auto_model.encode(clip_viz_dense.permute(0,3,1,2))

            output_path_lang = os.path.join(lang_feat_dir, f"{os.path.splitext(filename)[0]}_f.npy")
            output_path_lang3 = os.path.join(lang_feat_dim3, f"{os.path.splitext(filename)[0]}_f.npy")

            clip_viz_dense_dim3_resize = F.interpolate(clip_viz_dense_dim3, size=(height, width), mode='bilinear', align_corners=False)
            #clip_viz_dense_dim3 = clip_viz_dense_dim3[0].cpu().detach().numpy()
            #scale up the image
            cv2.imwrite(os.path.join(img_save_dir, filename), clip_viz_dense_dim3_resize[0].permute(1,2,0).cpu().detach().numpy()*255)

            #np.save(output_path_lang, clip_viz_dense[0].cpu().numpy())
            #np.save(output_path_lang3, clip_viz_dense_dim3[0].cpu().detach().numpy())

def perform_similarity(clip_viz_dense, text_embs):
    sims = clip_viz_dense @ text_embs.T
    sims = sims.squeeze()
    sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    return sim_norm

@torch.no_grad()
def speed_inference_test(aug, model):
    input_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test/results"

    times = []
    text_embs = get_user_embed()

    ckpt_path = f'auto_ckpt/replica_room0/best_ckpt.pth'
    checkpoint = torch.load(ckpt_path)

    encoder_hidden_dims = [384, 192, 96, 48, 24, 12, 3]
    decoder_hidden_dims = [12, 24, 48, 96, 192, 384, 768]
    auto_model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    auto_model.load_state_dict(checkpoint)
    auto_model.eval()

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            original_image = read_image(image_path, format="BGR")
            height, width = original_image.shape[:2]
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to("cuda")
            inputs = [{"image": image, "height": height, "width": width}]
            t1 = time.time()
            clip_viz_dense = get_lang_feat(inputs, model)
            clip_viz_dense_dim3 = auto_model.encode(clip_viz_dense.permute(0,3,1,2))
            clip_viz_recon = auto_model.decode(clip_viz_dense_dim3)
            clip_viz_recon = clip_viz_recon.permute(0,2,3,1)
            elapsed_time = (time.time() - t1)*1000.0
            sim_norm_clip = perform_similarity(clip_viz_recon, text_embs)
            cmap = plt.get_cmap("turbo")
            heatmap_clip = cmap(sim_norm_clip.detach().cpu().numpy())
            plt.imsave("/media/vision/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/rgb_heatmap/" + filename, heatmap_clip)
            print(f"Inference time for {filename}: {elapsed_time} ms")
            times.append(elapsed_time)


    print(f"Average inference time: {np.mean(times[1:])} ms")

@torch.no_grad()
def test_gaussian_lang_feat():
        # height, width = original_image.shape[:2]
        # image = aug.get_transform(original_image).apply_image(original_image)
        # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        # image.to("cuda")
        # inputs = [{"image": image, "height": height, "width": width}]
        # _, dense_clip_viz = model(inputs)
        # print("dense_clip_viz shape: ", dense_clip_viz.shape)
        # clip_viz_dense = dense_clip_viz.permute(0,2,3,1)
        
        clip_viz_dense_dim3 =  np.load("results/datasets_Replica/langsplat_labels/psnr/before_opt/lang/frame_000001.npy")
        clip_viz_dense_dim3 = torch.from_numpy(clip_viz_dense_dim3).float().to("cuda").unsqueeze(0)
        #sem_feat_resize = F.interpolate(clip_viz_dense_dim3.permute(0,3,1,2), size=(24, 24), mode='bilinear', align_corners=False)
        #restored_feat = model.decode(sem_feat_resize)
        #restored_feat_resize = F.interpolate(restored_feat, size=(h, w), mode='bilinear', align_corners=False).permute(0,2,3,1)
            
        ckpt_path = f'/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test/langsplat/auto_encoder/room0_langsplat_slam_convnext_large_d_320/best_ckpt.pth'
        checkpoint = torch.load(ckpt_path)
        encoder_hidden_dims = [384, 192, 96, 48, 24, 12, 3]
        decoder_hidden_dims = [12, 24, 48, 96, 192, 384, 768]
        auto_model = AutoencoderLangsplat(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
        auto_model.load_state_dict(checkpoint)
        auto_model.eval()
        clip_viz_recon = auto_model.decode(clip_viz_dense_dim3)
        #clip_viz_recon = clip_viz_recon.permute(0,2,3,1)
        
        name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
        tokenizer = open_clip.get_tokenizer(name)
        texts = tokenizer(["vase"]).cuda()
        print("texts shape: ", texts.shape)
        clip_model, _, _ = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device="cuda",)
        text_embs = clip_model.encode_text(texts)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        print("text_embs shape: ", text_embs.shape)

        sims = clip_viz_recon @ text_embs.T
        sims = sims.squeeze()
        sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("turbo")
        heatmap = cmap(sim_norm.cpu().numpy())
        plt.imshow(heatmap)
        plt.show()

@torch.no_grad()
def test_model_speed(model, aug):
    #input_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test/results"

    #times = []

    import glob
    image_paths = sorted(glob.glob("/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test/results/frame*.jpg"))
    
    #model_large = torch.load("/home/rpng/Documents/sai_ws/gripper_slam/latest/GaussianGripMapping/language/sed_model_large.pth")
    model.eval()
    avg_time = 0

    for img_path in image_paths:
        original_image = read_image(img_path, format="BGR")
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to("cuda")
        inputs = [{"image": image, "height": height, "width": width}]
        t1 = time.time()
        _, clip_features = model(inputs)
        end_time = time.time() - t1
        print(f"Time taken for {img_path}: {end_time}")
        avg_time += end_time
    
    print(f"Average time taken: {avg_time/len(image_paths)}")
    
if __name__ == "__main__":
    #test_gaussian_lang_feat()

    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    t1 = time.time()
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model_state_path = "seg_clip_model_l.pth"
    torch.save(model, model_state_path)
    print(f"Model building time: {(time.time() - t1)*1000.0} ms")


    # model_path="/home/vision/Documents/GaussianGripMapping/seg_clip_model_l.pth"
    # t1 = time.time()
    # model = load_lang_model(model_path=model_path)
    # print(f"Model loading time: {(time.time() - t1)} s")
    # print("Model created")

    # aug = T.ResizeShortestEdge(
    #     [640, 640], 2560
    # )
    # model.to("cuda")
    # test_model_speed(model, aug)

    #onnx_speed_test(aug)
    #optimize_onnx()

    # #input_format = cfg.INPUT.FORMAT
    # model.to("cuda")
    # save_npy_files(aug, model)
    #speed_inference_test(aug, model)

    # input_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/rgb"
    # output_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/language_feat"

    # # Ensure the output directory exists
    # os.makedirs(output_dir, exist_ok=True)

    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         image_path = os.path.join(input_dir, filename)
    #         original_image = read_image(image_path, format="BGR")
    #         height, width = original_image.shape[:2]
    #         image = aug.get_transform(original_image).apply_image(original_image)
    #         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    #         image.to("cuda")
    #         inputs = [{"image": image, "height": height, "width": width}]
    #         clip_viz_dense = get_lang_feat(inputs, model)
    #         output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_f.npy")
    #         np.save(output_path, clip_viz_dense[0].cpu().numpy())

    # with torch.no_grad():
    #     original_image = read_image("/media/vision/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/rgb/rgb_141.png", format="BGR")
    #     if input_format == "RGB":
    #         original_image = original_image[:, :, ::-1]
        
    #     height, width = original_image.shape[:2]
    #     image = aug.get_transform(original_image).apply_image(original_image)
    #     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    #     image.to("cuda")
    #     inputs = [{"image": image, "height": height, "width": width}]
    #     _, dense_clip_viz = model(inputs)
    #     print("dense_clip_viz shape: ", dense_clip_viz.shape)
    #     clip_viz_dense = dense_clip_viz.permute(0,2,3,1)
    #     name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    #     tokenizer = open_clip.get_tokenizer(name)
    #     texts = tokenizer(["window blinds"]).cuda()
    #     print("texts shape: ", texts.shape)
    #     clip_model, _, _ = open_clip.create_model_and_transforms(
    #             name,
    #             pretrained=pretrain,
    #             device="cuda",)
    #     text_embs = clip_model.encode_text(texts)
    #     text_embs /= text_embs.norm(dim=-1, keepdim=True)
    #     print("text_embs shape: ", text_embs.shape)

    #     sims = clip_viz_dense @ text_embs.T
    #     sims = sims.squeeze()
    #     sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    #     import matplotlib.pyplot as plt
    #     cmap = plt.get_cmap("turbo")
    #     heatmap = cmap(sim_norm.cpu().numpy())
    #     plt.imshow(heatmap)
    #     #plt.savefig("/home/vision/Documents/test_clip/heatmap.png")
    #     plt.show()

