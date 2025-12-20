import matplotlib.pyplot as plt
import numpy as np
import torch

def show_single_image(img, pred_attrs, true_attrs, ig_mask, cam_mask, attribute_names, show_ig_spatial=True):
    img_np = img.squeeze().permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0,0].imshow(img_np)
    axs[0,0].set_title("Original")
    axs[0,0].axis('off')
    axs[0,1].imshow(img_np)
    axs[0,1].set_title(f"Predicted: {np.array(attribute_names)[pred_attrs.bool().cpu().numpy()]}")
    axs[0,1].axis('off')
    axs[0,2].imshow(img_np)
    axs[0,2].set_title(f"Ground Truth: {np.array(attribute_names)[true_attrs.bool().cpu().numpy()]}")
    axs[0,2].axis('off')
    if show_ig_spatial:
        axs[1,0].imshow(ig_mask.squeeze().cpu().numpy(), cmap='hot')
        axs[1,0].set_title("IG Heatmap")
        axs[1,0].axis('off')
        axs[1,1].imshow(img_np)
        axs[1,1].imshow(ig_mask.squeeze().cpu().numpy(), cmap='jet', alpha=0.5)
        axs[1,1].set_title("IG Overlay")
        axs[1,1].axis('off')
    else:
        axs[1,0].bar(attribute_names, ig_mask.squeeze().cpu().numpy())
        axs[1,0].set_title("IG Per-Attribute")
        axs[1,0].tick_params(axis='x', rotation=90)
        axs[1,1].axis('off')
    axs[1,2].imshow(cam_mask.squeeze().cpu().numpy(), cmap='hot')
    axs[1,2].set_title("CAM++ Overlay")
    axs[1,2].axis('off')
    plt.tight_layout()
    plt.show()
