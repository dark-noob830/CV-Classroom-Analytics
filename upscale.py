import cv2
import os
from tqdm import tqdm
import json


class ImageUpscaler:
    def __init__(self, model_dir="models", model_type="espcn"):
        """
        A class for managing image super-resolution.

        Parameters:
        - model_dir: Path to the folder containing models.
        - model_type: Type of model (espcn, fsrcnn, lapsrn, edsr).
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.models = {}  # Store loaded models


    def load_model(self, scale_factor):
        """
        Load the model for a specific scale factor.
        """
        model_path = os.path.join(self.model_dir, f"{self.model_type}_x{scale_factor}.pb")

        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Please download the model first.")
            return None

        if scale_factor not in self.models:
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(model_path)
                sr.setModel(self.model_type, scale_factor)
                self.models[scale_factor] = sr
                print(f"\nModel {model_path} successfully loaded.")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                return None

        return self.models[scale_factor]

    def upscale_image(self, img, target_size=(160, 160)):
        """
        Upscale the image using super-resolution and resize to the target size.

        Parameters:
        - img: Input image.
        - target_size: Target size (width, height).
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # If the image is smaller than the target size
        if w < target_w or h < target_h:
            # Calculate the required upscale factor
            scale_factor = max(target_w / w, target_h / h)

            # Select the nearest available factor (2, 3, or 4)
            if scale_factor <= 2:
                selected_factor = 2
            elif scale_factor <= 3:
                selected_factor = 3
            else:
                selected_factor = 4

            # Load the required model
            model = self.load_model(selected_factor)
            if model is None:
                # If model not available, use regular resizing
                print(f"No model available for factor {selected_factor}. Using regular resizing.")
                return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

            # Apply super-resolution
            upscaled = model.upsample(img)

            # Resize to the exact target size
            return cv2.resize(upscaled, target_size, interpolation=cv2.INTER_LANCZOS4)

        # If the image is larger than or equal to the target size
        else:
            return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    def process_from_json(self, json_path, output_dir, target_size=(160, 160)):
        """
        Process images listed in a JSON file.

        Parameters:
        - json_path: Path to the JSON file containing image info.
        - output_dir: Folder to save processed images.
        - target_size: Target size for images.
        """

        # Load JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Found {len(data)} images in JSON to process...")

        for item in tqdm(data, desc="Processing images from JSON"):
            input_path = item['filename']
            output_path = os.path.join(output_dir, os.path.basename(input_path))

            try:
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Error reading image: {input_path}")
                    continue

                processed_img = self.upscale_image(img, target_size)
                cv2.imwrite(output_path, processed_img)

            except Exception as e:
                print(f"Error processing image {input_path}: {e}")


# Example usage
if __name__ == "__main__":
    # Directories
    json_path = "./data_extract/face_metadata.json"
    output_directory = "./data_extract/processed_faces"

    # Create an instance of the class
    upscaler = ImageUpscaler(model_dir="./models", model_type="edsr")

    # Process all images in the input folder
    upscaler.process_from_json(
        json_path=json_path,
        output_dir=output_directory,
        target_size=(160, 160)
    )

    print("All images processed successfully!")
