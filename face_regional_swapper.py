
import cv2
import numpy as np
import os
import argparse
from modules.face_analyser import get_one_face
from modules.processors.frame.face_swapper import swap_face, get_face_swapper
import modules.globals

def create_feature_mask(image_shape, landmarks, feature_indices):
    """
    Creates a mask for a specific facial feature.

    Args:
        image_shape: The shape of the image.
        landmarks: The facial landmarks.
        feature_indices: The indices of the landmarks for the feature.

    Returns:
        The mask for the feature.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    feature_points = landmarks[feature_indices].astype(np.int32)
    cv2.fillConvexPoly(mask, feature_points, 255)
    return mask

def regional_swap(source_image_path: str, target_image_path: str, output_path: str):
    """
    Performs a regional face swap.

    Args:
        source_image_path: Path to the source image.
        target_image_path: Path to the target image.
        output_path: Path to save the output image.
    """

    # Load the images
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)

    if source_image is None:
        print(f"Error: Could not read image from {source_image_path}")
        return
    if target_image is None:
        print(f"Error: Could not read image from {target_image_path}")
        return

    # Get the face objects
    source_face = get_one_face(source_image)
    target_face = get_one_face(target_image)

    if not source_face:
        print(f"No face detected in the source image: {source_image_path}")
        return
    if not target_face:
        print(f"No face detected in the target image: {target_image_path}")
        return

    # Get the face swapper model
    face_swapper = get_face_swapper()
    
    # Set the execution provider
    modules.globals.execution_providers = ["CPUExecutionProvider"]

    # Perform the two-way swaps
    swapped_image_1 = swap_face(source_face, target_face, target_image.copy()) # source on target
    swapped_image_2 = swap_face(target_face, source_face, source_image.copy()) # target on source

    # Define the facial regions (example, you might need to adjust the indices)
    # These indices are based on the 106-point landmark model from insightface
    eye_indices = list(range(33, 42)) + list(range(87, 96))
    nose_indices = list(range(52, 61))
    mouth_indices = list(range(61, 87))

    # Create the masks
    eye_mask = create_feature_mask(target_image.shape, target_face.landmark_2d_106, eye_indices)
    nose_mask = create_feature_mask(target_image.shape, target_face.landmark_2d_106, nose_indices)
    mouth_mask = create_feature_mask(target_image.shape, target_face.landmark_2d_106, mouth_indices)

    # Resize swapped_image_2 to match the target image size
    swapped_image_2_resized = cv2.resize(swapped_image_2, (target_image.shape[1], target_image.shape[0]))

    # Combine the swapped images using the masks
    final_image = target_image.copy()
    final_image[eye_mask > 0] = swapped_image_1[eye_mask > 0]
    final_image[nose_mask > 0] = swapped_image_2_resized[nose_mask > 0]
    final_image[mouth_mask > 0] = swapped_image_1[mouth_mask > 0]

    # Save the output image
    cv2.imwrite(output_path, final_image)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a regional face swap.")
    parser.add_argument("father_image", help="Path to the father's image.")
    parser.add_argument("mother_image", help="Path to the mother's image.")
    parser.add_argument("output_dir", help="Directory to save the output images.")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Create the son image
    son_output_path = os.path.join(args.output_dir, "son.jpg")
    regional_swap(args.mother_image, args.father_image, son_output_path)

    # Create the daughter image
    daughter_output_path = os.path.join(args.output_dir, "daughter.jpg")
    regional_swap(args.father_image, args.mother_image, daughter_output_path)
