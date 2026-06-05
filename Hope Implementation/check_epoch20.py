import os
import glob
import pandas as pd

def check_epoch_20_vs_best(base_dir):
    history_files = glob.glob(os.path.join(base_dir, "**", "history.csv"), recursive=True)
    
    if not history_files:
        print("No history.csv files found in the specified directory.")
        return

    print(f"{'Variant/Fold':<40} | {'Best Epoch':<12} | {'Best Val Acc':<15} | {'Epoch 20 Val Acc':<15}")
    print("-" * 90)

    for file_path in sorted(history_files):
        folder_name = os.path.basename(os.path.dirname(file_path))
        df = pd.read_csv(file_path)
        
        # Find best overall
        best_row = df.loc[df['val_acc'].idxmax()]
        best_epoch = int(best_row['epoch'])
        best_acc = best_row['val_acc']
        
        # Find epoch 20
        epoch_20_row = df[df['epoch'] == 20]
        if not epoch_20_row.empty:
            epoch_20_acc = epoch_20_row.iloc[0]['val_acc']
        else:
            epoch_20_acc = "N/A (Did not reach)"
            
        print(f"{folder_name:<40} | {best_epoch:<12} | {best_acc:<15.4f} | {epoch_20_acc:<15.4f}")

if __name__ == "__main__":
    check_epoch_20_vs_best("/Users/khoale/Downloads/kaggle/working/coding_space/Hope Implementation/checkpoints/")
