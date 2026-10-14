# Place trained model weights here.
# Expected filename: best_model.pth
#
# The checkpoint file should be a PyTorch state-dict saved as:
#
#   torch.save({
#       "epoch": epoch,
#       "val_acc": val_acc,
#       "model_state_dict": model.state_dict(),
#       "optimizer_state_dict": optimizer.state_dict(),
#   }, "checkpoints/best_model.pth")
#
# The app.py dashboard will automatically load this file if it exists.
# Without a checkpoint, the model will run with ImageNet-pretrained weights only
# (random classifier head), so predictions will be meaningless until training.
#
# Download pre-trained weights or run:
#   python train.py --data_dir data/ --output_dir checkpoints/ --epochs 20
