
from src.data_loader import load_fashion_mnist

data = load_fashion_mnist()

print("\n========== DATASET SUMMARY ==========")

for key, value in data.items():
    if key != "class_names":
        print(f"{key}: {value.shape}")

print("\nClass names:", data["class_names"])
print("Pixel range:", data["X_train_cnn"].min(),
      "to", data["X_train_cnn"].max())

print("\nDataset loaded successfully!")
