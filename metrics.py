from model import *
import numpy as np
import seaborn as sns
import tqdm
from pathlib import Path
import argparse
from image_feeder import read_labels_from_ubyte, read_images_from_ubyte

# 修改评价方法
def calculate_metrics(predict_recall, images: np.ndarray, labels: np.ndarray|list[int], max_test: int | None = None):
    confusion_matrix = np.zeros((10, 10), dtype=np.float32)
    correct = 0
    total = 0
    n = images.shape[0] if max_test is None else min(images.shape[0], max_test)
    for idx in tqdm.tqdm(range(n), desc="test", leave=False):
        image = images[idx]
        label = int(labels[idx])
        pred = predict_recall(image)
        confusion_matrix[pred,label] += 1
        correct += int(pred == label)
        total += 1

    accuracy = correct / max(total, 1)
    for i in range(10):
        col_sum = confusion_matrix[:, i].sum()
        if col_sum > 0:
            confusion_matrix[:, i] = confusion_matrix[:, i] / col_sum
            
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.sum(recall) / 10.0
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:\n{confusion_matrix}")

    return accuracy,recall , confusion_matrix


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate the model on the test set.")
    argparser.add_argument("--model", type=str,default = "./full_trained/lenet5_best_epoch37_acc9500.pkl", help="Path to the trained model.")
    argparser.add_argument("--test_data_path", type=str,default="./dataset" , help="Directory containing the test data.")
    args = argparser.parse_args()
    
    test_labels = read_labels_from_ubyte(f"{args.test_data_path}/t10k-labels-idx1-ubyte")
    test_images = read_images_from_ubyte(f"{args.test_data_path}/t10k-images-idx3-ubyte")

    # test_labels = test_labels[:1000] 
    # test_images = test_images[:1000] 

    test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    test_images = test_images.astype("float32")/127.5 - 1.0

    model = Lenet5.load_model(args.model)
    accuracy, recall, confusion_matrix = calculate_metrics(model.predict_label, test_images, test_labels)
    # 学一下yolo的作图
    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
    confusion_matrix[confusion_matrix < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8},
                 cmap='Blues', fmt='.2f', square=True,).set_facecolor((1, 1, 1))
    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    # fig.show()
    fig.savefig(Path("./images") / 'confusion_matrix.png', dpi=196)
    plt.show()
