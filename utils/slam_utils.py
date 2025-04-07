import torch
import torch.nn.functional as F

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err

def extract_edges(feature_map, threshold=None):
    # Define Sobel kernels
    sobel_kernel_x = torch.tensor([[[[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]]], dtype=torch.float32)
    sobel_kernel_y = torch.tensor([[[[-1, -2, -1],
                                     [ 0,  0,  0],
                                     [ 1,  2,  1]]]], dtype=torch.float32)

    # Move kernels to the same device as the feature_map
    sobel_kernel_x = sobel_kernel_x.to(feature_map.device)
    sobel_kernel_y = sobel_kernel_y.to(feature_map.device)

    # Repeat kernels for each channel in the input feature map
    channels = feature_map.shape[1]
    sobel_kernel_x = sobel_kernel_x.repeat(channels, 1, 1, 1)  # Shape: [C, 1, 3, 3]
    sobel_kernel_y = sobel_kernel_y.repeat(channels, 1, 1, 1)  # Shape: [C, 1, 3, 3]

    # Apply convolution to compute gradients
    grad_x = F.conv2d(feature_map, sobel_kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(feature_map, sobel_kernel_y, padding=1, groups=channels)

    # Compute edge magnitude
    edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    
    # Normalize the edges between [0, 1] channel-wise
    edges_min = edges.view(channels, -1).min(dim=1)[0].view(channels, 1, 1, 1)
    edges_max = edges.view(channels, -1).max(dim=1)[0].view(channels, 1, 1, 1)
    edges = (edges - edges_min) / (edges_max - edges_min + 1e-6)
    
    # If threshold is provided, apply it
    if threshold is not None:
        edges = torch.where(edges > threshold, edges, torch.tensor(0.0).to(edges.device))
    
    return edges

def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)

def compute_gradients(feature_map):
    grad_x = torch.abs(feature_map[:, :, :-1] - feature_map[:, :, 1:])
    grad_y = torch.abs(feature_map[:, :-1, :] - feature_map[:, 1:, :])
    return grad_x, grad_y

def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    # #also add edge smoothing loss this over smooths the image/adds smudge effect
    # ren_img_grad_x, ren_img_grad_y = compute_gradients(image)
    # gt_image_grad_x, gt_image_grad_y = compute_gradients(gt_image)

    # #weights based on image gradients
    # weight_x = torch.exp(-torch.mean(gt_image_grad_x, dim=1, keepdim=True))
    # weight_y = torch.exp(-torch.mean(gt_image_grad_y, dim=1, keepdim=True))

    # smoothness_loss_x = (ren_img_grad_x * weight_x).mean()
    # smoothness_loss_y = (ren_img_grad_y * weight_y).mean()

    # smoothness_loss = smoothness_loss_x + smoothness_loss_y
    # tv_loss = torch.mean(torch.abs(image[:, :, :-1] - image[:, :, 1:])) 
    # + torch.mean(torch.abs(image[:, :-1, :] - image[:, 1:, :]))
    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean() #+ 0.01 * tv_loss


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
