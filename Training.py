import cv2
import numpy as np
import os
import pickle

def train_model(people):
    print("\nTraining face recognition model for multiple people...")
    
    # Dictionary to store label mapping
    label_mapping = {}
    
    Training_Data, Labels = [], []
    image_size = (200, 200)
    
    for idx, person in enumerate(people):
        # Map person name to numeric label
        label_mapping[idx] = person
        data_path = f'F:/Kanishak/IPU/IPU/Face_Rec/dataset/processed_{person}/'
        
        if not os.path.exists(data_path):
            print(f"Warning: Directory for {person} not found at {data_path}.")
            print(f"         Did you run the face extraction script for {person}?")
            continue
            
        # Only use grayscale processed images
        files = [f for f in os.listdir(data_path) if f.endswith('_gray.jpg')]
        
        if not files:
            print(f"Warning: No processed images found for {person}.")
            continue
            
        print(f"Found {len(files)} images for {person} (Label: {idx})")
        
        for file in files:
            image_path = os.path.join(data_path, file)
            # Load directly as grayscale to be consistent
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
            if image is None:
                print(f"Skipping {file} (Invalid Image)")
                continue
    
            # Ensure size is consistent
            if image.shape != image_size:
                image = cv2.resize(image, image_size)
            
            # Add to training data with the current label (person index)
            Training_Data.append(np.asarray(image, dtype=np.uint8))
            Labels.append(idx)
    
    if not Training_Data:
        print("Error: No valid images found for any person. Cannot train model.")
        return False
        
    print(f"Prepared total of {len(Training_Data)} images for training.")
    
    # Convert to numpy arrays
    Training_Data = np.asarray(Training_Data)
    Labels = np.asarray(Labels, dtype=np.int32)
    
    # Create and train the model with adjusted parameters
    model = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=10,
        grid_y=6,
        threshold=100
    )
    
    # Train the model
    model.train(Training_Data, Labels)
    
    # Save the trained model
    model.save("face_recognition_model.xml")
    
    # Save the label mapping dictionary
    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(label_mapping, f)
    
    print("✅ Model training completed!")
    print(f"Saved model as 'face_recognition_model.xml'")
    print(f"Saved label mapping as 'label_mapping.pkl'")
    
    # Test on training data
    correct = 0
    print("\nTesting on training data:")
    for i, face in enumerate(Training_Data):
        label, confidence = model.predict(face)
        predicted_confidence = 100 * (1 - confidence / 300)
        person_name = label_mapping[label]
        expected_name = label_mapping[Labels[i]]
        
        result = "✓" if label == Labels[i] else "✗"
        print(f"Image {i+1}: Predicted {person_name} ({predicted_confidence:.1f}%), Expected {expected_name} {result}")
        
        if label == Labels[i]:
            correct += 1
    
    # accuracy = 100 * correct / len(Training_Data)
    # print(f"\nTraining accuracy: {correct}/{len(Training_Data)} ({accuracy:.2f}%)")
    
    return True

def main():
    print("Multi-Person Face Recognition Trainer")
    print("====================================")
    
    people = []
    while True:
        person = input("Enter person's name (or 'done' to finish): ")
        if person.lower() == 'done':
            break
        people.append(person)
    
    if not people:
        print("No people specified. Exiting.")
        return
    
    print(f"\nWill train model for: {', '.join(people)}")
    train_model(people)

if __name__ == "__main__":
    main()