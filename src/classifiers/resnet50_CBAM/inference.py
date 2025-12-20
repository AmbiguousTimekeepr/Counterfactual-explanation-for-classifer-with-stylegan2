"""
Inference utilities for the classifier
"""
import torch
import numpy as np
from PIL import Image
import os


def inference_single_image(model, image_path, transform, attribute_names, device, 
                          df_attr=None, threshold=0.5):
    """
    Inference cho 1 ảnh, in ra ground truth, prediction probability và accuracy
    
    Args:
        model: Trained model
        image_path: Đường dẫn đến ảnh cần test
        transform: Transform để preprocess ảnh
        attribute_names: List tên các attributes
        device: Device để chạy inference
        df_attr: DataFrame chứa ground truth labels (optional)
        threshold: Ngưỡng để chuyển prob thành binary prediction (default=0.5)
    
    Returns:
        probs: Prediction probabilities
        binary_preds: Binary predictions
        gt_labels: Ground truth labels (if available)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Get ground truth label from CSV
    img_name = os.path.basename(image_path)
    if df_attr is not None and img_name in df_attr.index:
        gt_labels = df_attr.loc[img_name].values
        gt_labels = np.where(gt_labels == -1, 0, gt_labels)  # Convert -1 to 0
    else:
        gt_labels = None
        if df_attr is not None:
            print(f"Warning: {img_name} not found in CSV. Cannot get ground truth.")
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]  # Shape: (40,)
    
    # Convert probabilities to binary predictions
    binary_preds = (probs > threshold).astype(int)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Inference Result for: {img_name}")
    print(f"{'='*80}")
    print(f"{'Attribute':<25} | {'Ground Truth':<13} | {'Pred Prob':<10} | {'Prediction':<11} | {'Correct':<7}")
    print(f"{'-'*80}")
    
    correct_count = 0
    total_count = len(attribute_names)
    
    for i, attr_name in enumerate(attribute_names):
        pred_prob = probs[i]
        pred_label = binary_preds[i]
        
        if gt_labels is not None:
            gt_label = int(gt_labels[i])
            is_correct = (pred_label == gt_label)
            correct_count += int(is_correct)
            correct_str = "✓" if is_correct else "✗"
            gt_str = str(gt_label)
        else:
            gt_str = "N/A"
            correct_str = "N/A"
        
        print(f"{attr_name:<25} | {gt_str:<13} | {pred_prob:.4f}     | {pred_label:<11} | {correct_str:<7}")
    
    print(f"{'-'*80}")
    
    if gt_labels is not None:
        accuracy = correct_count / total_count
        print(f"\nImage Accuracy: {correct_count}/{total_count} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print("\nNo ground truth available - cannot calculate accuracy")
    
    print(f"{'='*80}\n")
    
    return probs, binary_preds, gt_labels
