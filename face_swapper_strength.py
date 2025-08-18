
import cv2
import numpy as np
import os
import argparse
from modules.face_analyser import get_one_face
from modules.processors.frame.face_swapper import swap_face, get_face_swapper
import modules.globals

def swap_with_strength(source_image_path: str, target_image_path: str, output_path: str, strength: float):
    """
    Swaps a face from a source image to a target image with a specified strength.

    Args:
        source_image_path: Path to the source image.
        target_image_path: Path to the target image.
        output_path: Path to save the output image.
        strength: The strength of the swap (0.0 to 1.0).
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

    # Perform the face swap
    swapped_image = swap_face(source_face, target_face, target_image.copy())

    # Alpha blend the swapped image with the original target image
    final_image = (swapped_image * strength + target_image * (1 - strength)).astype(np.uint8)

    # Save the output image
    cv2.imwrite(output_path, final_image)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swap faces with a specified strength.")
    parser.add_argument("father_image", help="Path to the father's image.")
    parser.add_argument("mother_image", help="Path to the mother's image.")
    parser.add_argument("output_dir", help="Directory to save the output images.")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Create the son image
    son_output_path = os.path.join(args.output_dir, "son.jpg")
    swap_with_strength(args.mother_image, args.father_image, son_output_path, 0.5)

    # Create the daughter image
    daughter_output_path = os.path.join(args.output_dir, "daughter.jpg")
    swap_with_strength(args.father_image, args.mother_image, daughter_output_path, 0.5)
