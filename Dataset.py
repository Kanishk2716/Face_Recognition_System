import cv2
import os
import sys

# Load the Haar Cascade
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 8)

    if len(faces) == 0:
        return None

    # Get the largest face (assuming the main face is the largest one)
    largest_face = None
    largest_area = 0
    
    for (x, y, w, h) in faces:
        face_area = w * h
        if face_area > largest_area:
            largest_area = face_area
            largest_face = img[y:y+h, x:x+w]
    
    return largest_face

def process_person(person_name):
    # Folder containing images
    input_folder = f"F:/Kanishak/IPU/IPU/Face_Rec/dataset/{person_name}/"
    output_folder = f"F:/Kanishak/IPU/IPU/Face_Rec/dataset/processed_{person_name}/"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Check if input directory exists
    if not os.path.exists(input_folder):
        print(f"Error: Input directory for {person_name} does not exist at {input_folder}")
        return False

    count = 0
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    for file in os.listdir(input_folder):
        if not file.lower().endswith(valid_extensions):
            print(f"Skipping {file} (Not an image file)")
            continue
            
        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {file} (Invalid Image)")
            continue

        face = face_extractor(img)
        if face is not None:
            count += 1
            # Keep consistent preprocessing: resize then convert to grayscale
            face = cv2.resize(face, (200, 200))
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Save both color and grayscale versions for debugging
            gray_file_path = os.path.join(output_folder, f"face_{count}_gray.jpg")
            color_file_path = os.path.join(output_folder, f"face_{count}_color.jpg")
            
            cv2.imwrite(gray_file_path, face_gray)
            cv2.imwrite(color_file_path, face)  # Save color version for reference

            print(f"Processed: {file} → {gray_file_path}")
        else:
            print(f"No face found in {file}")

    print(f"✅ Face Extraction Completed for {person_name}! Processed {count} images.")
    return count > 0

def main():
    print("Multi-Person Face Extraction Tool")
    print("=================================")
    
    people = []
    while True:
        person = input("Enter person's name (or 'done' to finish): ")
        if person.lower() == 'done':
            break
        people.append(person)
    
    if not people:
        print("No people specified. Exiting.")
        return
    
    print(f"\nWill process faces for: {', '.join(people)}")
    
    success_count = 0
    for person in people:
        print(f"\nProcessing images for {person}...")
        if process_person(person):
            success_count += 1
    
    print(f"\nCompleted processing for {success_count}/{len(people)} people.")
    print("You can now run the training script.")

if __name__ == "__main__":
    main()