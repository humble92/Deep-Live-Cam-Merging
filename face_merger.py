
import cv2
import numpy as np
import os
import argparse
from modules.face_analyser import get_one_face
from modules.typing import Face
from modules.processors.frame.face_swapper import swap_face, get_face_swapper
import modules.globals

def merge_faces(father_image_path: str, mother_image_path: str, output_dir: str):
    """
    Merges the faces of a man and a woman to create "son" and "daughter" images.

    Args:
        father_image_path: Path to the father's image.
        mother_image_path: Path to the mother's image.
        output_dir: Directory to save the output images.
    """

    # Load the parent images
    father_image = cv2.imread(father_image_path)
    mother_image = cv2.imread(mother_image_path)

    if father_image is None:
        print(f"Error: Could not read image from {father_image_path}")
        return
    if mother_image is None:
        print(f"Error: Could not read image from {mother_image_path}")
        return

    # Get the face objects for the parents
    father_face = get_one_face(father_image)
    mother_face = get_one_face(mother_image)

    if not father_face:
        print("No face detected in the father's image.")
        return
    if not mother_face:
        print("No face detected in the mother's image.")
        return

    # Get the face swapper model
    face_swapper = get_face_swapper()
    
    # Set the execution provider
    modules.globals.execution_providers = ["CPUExecutionProvider"]


    # --- Create the "son" ---
    son_embedding = (father_face.normed_embedding * 0.6) + (mother_face.normed_embedding * 0.4)
    son_face = Face(
        bounding_box=father_face.bounding_box,
        landmark_2d_106=father_face.landmark_2d_106,
        landmark_3d_68=father_face.landmark_3d_68,
        kps=father_face.kps,
        score=father_face.score,
        embedding=son_embedding,
        gender=1,  # 1 for male
        age=25,
    )
    
    # Use the father's image as the target for the son
    son_image = swap_face(son_face, father_face, father_image.copy())
    son_output_path = os.path.join(output_dir, "son.jpg")
    cv2.imwrite(son_output_path, son_image)
    print(f"Son image saved to {son_output_path}")

    # --- Create the "daughter" ---
    daughter_embedding = (father_face.normed_embedding * 0.6) + (mother_face.normed_embedding * 0.4)
    daughter_face = Face(
        bounding_box=mother_face.bounding_box,
        landmark_2d_106=mother_face.landmark_2d_106,
        landmark_3d_68=mother_face.landmark_3d_68,
        kps=mother_face.kps,
        score=mother_face.score,
        embedding=daughter_embedding,
        gender=0,  # 0 for female
        age=25,
    )

    # Use the mother's image as the target for the daughter
    daughter_image = swap_face(daughter_face, mother_face, mother_image.copy())
    daughter_output_path = os.path.join(output_dir, "daughter.jpg")
    cv2.imwrite(daughter_output_path, daughter_image)
    print(f"Daughter image saved to {daughter_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two faces to create 'son' and 'daughter' images.")
    parser.add_argument("father_image", help="Path to the father's image.")
    parser.add_argument("mother_image", help="Path to the mother's image.")
    parser.add_argument("output_dir", help="Directory to save the output images.")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    merge_faces(args.father_image, args.mother_image, args.output_dir)
