import face_recognition
import os
import shutil
import cv2

# Paths
reference_image_path = '/workspaces/Quantitative-Finance/Face_Detection/me.jpg'
images_directory = '/workspaces/Quantitative-Finance/Face_Detection/images'
sorted_directory = '/workspaces/Quantitative-Finance/Face_Detection/sorted_images'
unsorted_directory = '/workspaces/Quantitative-Finance/Face_Detection/unsorted_images'

# Create sorted and unsorted directories if they don't exist
os.makedirs(sorted_directory, exist_ok=True)
os.makedirs(unsorted_directory, exist_ok=True)

# Load the reference image and encode the face
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Function to align and encode faces
def align_and_encode(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return face_encodings
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

# Function to compare faces with a higher tolerance
def is_face_match(image_path, reference_encoding, tolerance=0.6):
    unknown_encodings = align_and_encode(image_path)
    for unknown_encoding in unknown_encodings:
        results = face_recognition.compare_faces([reference_encoding], unknown_encoding, tolerance=tolerance)
        if results[0]:
            return True
    return False

# Iterate over images in the directory
for image_filename in os.listdir(images_directory):
    image_path = os.path.join(images_directory, image_filename)
    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        if is_face_match(image_path, reference_encoding, tolerance=0.6):  # Adjust tolerance as needed
            shutil.copy(image_path, os.path.join(sorted_directory, image_filename))
            print(f"Copied {image_filename} to {sorted_directory}")
        else:
            shutil.copy(image_path, os.path.join(unsorted_directory, image_filename))
            print(f"Copied {image_filename} to {unsorted_directory}")
