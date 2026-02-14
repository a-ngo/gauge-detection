"""Simple training script"""

from ultralytics import YOLO


def main():
    # Load a pretrained model
    model = YOLO("yolov8m.pt")

    # Train the model with MPS
    data_path = "../data/data.yaml"
    results = model.train(
        data=data_path,
        epochs=20,
        imgsz=640,
        device="mps",
        # Freeze backbone, only train head
        freeze=10,  # Freeze first 10 layers (backbone)
        # Light augmentation (already learned general features)
        # augment=True,
        # degrees=15,  # Less than full training
        # translate=0.1,
        # scale=0.3,
    )


if __name__ == "__main__":
    main()
