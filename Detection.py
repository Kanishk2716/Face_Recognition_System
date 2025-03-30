import cv2
import numpy as np
import os
import pickle

def load_model_and_labels():
    # Check if model file exists
    if not os.path.exists("face_recognition_model.xml"):
        print("Error: Model file 'face_recognition_model.xml' not found.")
        print("Please run the training script first.")
        return None, None
        
    # Check if label mapping file exists
    if not os.path.exists("label_mapping.pkl"):
        print("Error: Label mapping file 'label_mapping.pkl' not found.")
        print("Please run the training script first.")
        return None, None

    # Load the trained model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("face_recognition_model.xml")
    
    # Load the label mapping
    with open("label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)
    
    return model, label_mapping

def face_detector(img, draw_rectangle=True):
    # Load the Haar cascade for face detection
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 8)

    if len(faces) == 0:
        return img_copy, None

    # Find all faces in the image
    detected_faces = []
    
    for (x, y, w, h) in faces:
        if draw_rectangle:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        detected_faces.append((face_roi, (x, y, w, h)))
    
    return img_copy, detected_faces

def create_side_by_side_image(original, processed):
    # Resize both images to the required dimensions (613x378)
    original_resized = cv2.resize(original, (613, 378))
    processed_resized = cv2.resize(processed, (613, 378))
    
    # Create a blank canvas to hold both images side by side
    combined_width = 613 * 2  # Two images of width 613
    combined_img = np.zeros((378, combined_width, 3), dtype=np.uint8)
    
    # Place original image on the left
    combined_img[0:378, 0:613] = original_resized
    
    # Place processed image on the right
    combined_img[0:378, 613:613*2] = processed_resized
    
    # Add labels
    cv2.putText(combined_img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(combined_img, "Recognized", (613 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Add a dividing line
    cv2.line(combined_img, (613, 0), (613, 378), (255, 255, 255), 2)
    
    return combined_img

def test_image(model, label_mapping, image_path=None):
    # If no image path provided, prompt user
    if image_path is None or not os.path.exists(image_path):
        image_path = input("Enter path to test image: ")
        
    # Load test image
    test_image = cv2.imread(image_path)
    
    if test_image is None:
        print(f"Error: Image not found at {image_path}!")
        return
    
    # Create a copy of the original image before processing
    original_image = test_image.copy()
    
    # Create debug directory if needed
    debug_dir = "debug_output"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Detect faces in the image
    processed_image, detected_faces = face_detector(test_image)
    
    if not detected_faces:
        print("No faces detected in the image.")
        cv2.putText(processed_image, "No Faces Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Still create side-by-side display
        combined_image = create_side_by_side_image(original_image, processed_image)
        cv2.imshow('Face Recognition Results', combined_image)
        cv2.waitKey(0)
        return
    
    # Process each face
    for i, (face, face_coords) in enumerate(detected_faces):
        # Save detected face for debugging
        cv2.imwrite(os.path.join(debug_dir, f"test_face_{i+1}.jpg"), face)
        
        # Make prediction
        label, distance = model.predict(face)
        confidence = int(100 * (1 - (distance / 300)))
        
        # Get person name from label
        if label in label_mapping:
            person_name = label_mapping[label]
        else:
            person_name = "Unknown Label"
        
        print(f"Face {i+1}: Predicted {person_name} with {confidence}% confidence")
        
        # Add label to the image
        x, y, w, h = face_coords
        if confidence > 45:  # Threshold for recognition
            label_text = f"{person_name}: {confidence}%"
            cv2.putText(processed_image, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            label_text = f"Unknown: {confidence}%"
            cv2.putText(processed_image, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Create side-by-side display
    combined_image = create_side_by_side_image(original_image, processed_image)
    
    # Display the combined result
    cv2.imshow('Face Recognition Results', combined_image)
    
    # Save the combined image if needed
    cv2.imwrite(os.path.join(debug_dir, "combined_result.jpg"), combined_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def webcam_recognition(model, label_mapping):
    print("Starting webcam face recognition...")
    print("Press 'q' to quit")
    
    # Load the Haar cascade for face detection
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make a copy of the original frame
        original_frame = frame.copy()
        
        # Process frame (slightly modified for real-time performance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 8)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            face_roi = gray[y:y+h, x:x+w]
            try:
                # Resize for model
                face_roi = cv2.resize(face_roi, (200, 200))
                
                # Predict
                label, distance = model.predict(face_roi)
                confidence = int(100 * (1 - (distance / 300)))
                
                # Get person name
                if label in label_mapping and confidence > 45:
                    person_name = label_mapping[label]
                    label_text = f"{person_name}: {confidence}%"
                    cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Unknown: {confidence}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Create side-by-side display for webcam
        combined_frame = create_side_by_side_image(original_frame, frame)
        
        # Show the combined frame
        cv2.imshow('Webcam Face Recognition', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Multi-Person Face Recognition System")
    print("===================================")
    
    # Load model and label mapping
    model, label_mapping = load_model_and_labels()
    if model is None or label_mapping is None:
        return
    
    print(f"Model loaded successfully. Trained for {len(label_mapping)} people:")
    for label, name in label_mapping.items():
        print(f"  Label {label}: {name}")
    
    while True:
        print("\nOptions:")
        print("1. Test on an image")
        print("2. Test on webcam")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            image_path = input("Enter path to test image (or press Enter for prompt): ")
            test_image(model, label_mapping, image_path if image_path else None)
        elif choice == '2':
            webcam_recognition(model, label_mapping)
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()