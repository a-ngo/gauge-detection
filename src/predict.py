"""Simple inference scipt"""

from ultralytics import YOLO


def main():
    # Load a pretrained YOLO26n model
    checkpoint_path = "../checkpoints/best.pt"
    model = YOLO(checkpoint_path)

    # Define path to a test image file
    img_path = "../data/test/images/20220610_172141817_iOS_jpg.rf.0a272867eb042f9da8b584bdc780defb.jpg"

    # Run inference on the source
    results = model(img_path, save=True)  # list of Results objects


if __name__ == "__main__":
    main()
