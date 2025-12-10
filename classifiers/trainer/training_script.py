import torch
import os
import numpy as np
import tqdm

def classifier_training(model, train_loader, val_loader, train_dataset, 
                        val_dataset, criterion, optimizer, scheduler, attribute_names, epochs, 
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                        save_path = r"./outputs/cnn_classfier" 
                        ):
    # --- 4. TRAINING LOOP ---
    print("Starting training loop...")

    # Biến lưu lịch sử (nếu muốn vẽ biểu đồ sau này)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- VALIDATE ---
        model.eval()
        val_running_loss = 0.0
        
        # Lưu prediction và target để tính accuracy cho từng attribute
        all_preds = []
        all_targets = []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
        
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                
                # Tính xác suất (Sigmoid)
                probs = torch.sigmoid(outputs)
                
                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_dataset)
        history['val_loss'].append(epoch_val_loss)

        # --- TÍNH TOÁN METRICS CHO TỪNG ATTRIBUTE ---
        all_preds = np.concatenate(all_preds, axis=0)   # Shape: (N_val, 40)
        all_targets = np.concatenate(all_targets, axis=0) # Shape: (N_val, 40)
        
        # Chuyển xác suất thành nhãn binary (Threshold = 0.5)
        binary_preds = (all_preds > 0.5).astype(int)
        
        # Tính accuracy cho từng cột (từng attribute)
        # Correct prediction: (pred == target)
        correct_counts = np.sum(binary_preds == all_targets, axis=0)
        acc_per_attr = correct_counts / len(val_dataset)
        
        mean_acc = np.mean(acc_per_attr)

        # --- IN KẾT QUẢ ---
        print(f"\n--- Epoch {epoch} Report ---")
        print(f"LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Mean Val Acc: {mean_acc:.4f}")
        
        print("-" * 60)
        print(f"{'Attribute':<25} | {'Accuracy':<10} | {'Sample Prob (Mean)':<15}")
        print("-" * 60)
        
        # In chi tiết từng attribute
        for i, attr_name in enumerate(attribute_names):
            # Sample Prob Mean: Xác suất trung bình model dự đoán cho class này (để xem model có bị bias về 0 hay 1 ko)
            mean_prob = np.mean(all_preds[:, i])
            print(f"{attr_name:<25} | {acc_per_attr[i]:.4f}     | {mean_prob:.4f}")
        print("-" * 60)

        # --- SAVE CHECKPOINT ---
        # Lưu định kỳ mỗi 5 epoch
        if epoch % 5 == 0:
            ckpt_name = f"resnet50_cbam_epoch_{epoch}.pth"
            save_path = os.path.join(save_path, ckpt_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

        # Lưu best model (nếu cần)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))

    print("Training Complete.")