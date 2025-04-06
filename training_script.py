import os
import cv2
import numpy as np
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from kyc_form_detector import KYCFormTextDetection

def prepare_training_dataset(source_dir, output_dir, splits=(0.7, 0.2, 0.1)):
    """
    Prepare a training dataset from KYC form images
    
    Args:
        source_dir: Directory containing KYC form images and annotations
        output_dir: Directory to save the processed dataset
        splits: Train/validation/test splits
    """
    # Create directories
    os.makedirs(os.path.join(output_dir, 'train', 'form_fields'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'non_form_fields'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'form_fields'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'non_form_fields'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'form_fields'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'non_form_fields'), exist_ok=True)
    
    # Get image files and annotations
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Process each image
    for img_file in image_files:
        # Check if annotation file exists
        base_name = os.path.splitext(img_file)[0]
        annotation_file = os.path.join(source_dir, f"{base_name}.txt")
        
        if not os.path.exists(annotation_file):
            print(f"Annotation not found for {img_file}, skipping...")
            continue
        
        # Read image
        img_path = os.path.join(source_dir, img_file)
        image = cv2.imread(img_path)
        
        # Read annotations
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
        
        field_regions = []
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:  # [class_id, x, y, w, h]
                class_id = int(parts[0])
                x = float(parts[1]) * image.shape[1]
                y = float(parts[2]) * image.shape[0]
                w = float(parts[3]) * image.shape[1]
                h = float(parts[4]) * image.shape[0]
                
                if class_id == 0:  # Assuming 0 is for form field
                    field_regions.append((int(x), int(y), int(w), int(h)))
        
        # Extract form fields and non-form fields
        form_field_count = 0
        non_field_count = 0
        
        # Determine dataset split
        rand = np.random.random()
        if rand < splits[0]:
            target_dir = 'train'
        elif rand < splits[0] + splits[1]:
            target_dir = 'val'
        else:
            target_dir = 'test'
        
        # Extract form fields
        for i, (x, y, w, h) in enumerate(field_regions):
            field_img = image[y:y+h, x:x+w]
            
            # Save form field
            field_img_resized = cv2.resize(field_img, (64, 64))
            field_path = os.path.join(output_dir, target_dir, 'form_fields', f"{base_name}_field_{i}.jpg")
            cv2.imwrite(field_path, field_img_resized)
            form_field_count += 1
            
            # Generate negative samples (non-form fields)
            for _ in range(min(3, form_field_count)):  # Generate up to 3 negative samples per positive
                # Random region that doesn't overlap with form fields
                valid_region = False
                attempts = 0
                
                while not valid_region and attempts < 10:
                    attempts += 1
                    nx = np.random.randint(0, image.shape[1] - 64)
                    ny = np.random.randint(0, image.shape[0] - 64)
                    nw = np.random.randint(32, 64)
                    nh = np.random.randint(32, 64)
                    
                    # Check overlap with form fields
                    overlap = False
                    for fx, fy, fw, fh in field_regions:
                        if (nx < fx + fw and nx + nw > fx and 
                            ny < fy + fh and ny + nh > fy):
                            overlap = True
                            break
                    
                    if not overlap:
                        valid_region = True
                        non_field_img = image[ny:ny+nh, nx:nx+nw]
                        non_field_img_resized = cv2.resize(non_field_img, (64, 64))
                        non_field_path = os.path.join(output_dir, target_dir, 'non_form_fields', 
                                                     f"{base_name}_non_field_{non_field_count}.jpg")
                        cv2.imwrite(non_field_path, non_field_img_resized)
                        non_field_count += 1
        
        print(f"Processed {img_file}: {form_field_count} form fields, {non_field_count} non-form fields")


def generate_synthetic_data(output_dir, num_samples=500):
    """
    Generate synthetic KYC form data for training
    
    Args:
        output_dir: Directory to save the synthetic data
        num_samples: Number of synthetic samples to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define form templates
    templates = [
        {"size": (800, 1000), "bg_color": (255, 255, 255)},
        {"size": (1000, 1200), "bg_color": (245, 245, 245)},
        {"size": (850, 1100), "bg_color": (250, 250, 250)}
    ]
    
    # Define form field types with positions
    field_types = [
        {"name": "Name", "box_size": (300, 40)},
        {"name": "Date of Birth", "box_size": (200, 40)},
        {"name": "Address", "box_size": (350, 40)},
        {"name": "ID Number", "box_size": (250, 40)},
        {"name": "Phone", "box_size": (200, 40)},
        {"name": "Email", "box_size": (250, 40)}
    ]
    
    for i in range(num_samples):
        # Select random template
        template = templates[np.random.randint(0, len(templates))]
        
        # Create blank form using np.full to ensure proper memory layout
        form = np.full((template["size"][1], template["size"][0], 3), template["bg_color"], dtype=np.uint8)
        
        # Add form title
        title = "KNOW YOUR CUSTOMER FORM"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_size = cv2.getTextSize(title, font, 1, 2)[0]
        title_x = (form.shape[1] - title_size[0]) // 2
        cv2.putText(form, title, (title_x, 50), font, 1, (0, 0, 0), 2)
        
        # Draw horizontal line below title
        cv2.line(form, (50, 70), (form.shape[1]-50, 70), (0, 0, 0), 2)
        
        # Add form fields
        y_pos = 150
        field_annotations = []
        
        for field in field_types:
            # Add field label
            cv2.putText(form, field["name"] + ":", (100, y_pos), font, 0.7, (0, 0, 0), 1)
            
            # Add field box
            box_x = 300
            box_y = y_pos - 30
            box_w = field["box_size"][0]
            box_h = field["box_size"][1]
            
            cv2.rectangle(form, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), 1)
            
            # Store annotation
            field_annotations.append([0, (box_x + box_w/2)/form.shape[1], 
                                        (box_y + box_h/2)/form.shape[0], 
                                        box_w/form.shape[1], 
                                        box_h/form.shape[0]])
            
            y_pos += 100
        
        # Save synthetic form
        form_path = os.path.join(output_dir, f"syn_form_{i}.jpg")
        cv2.imwrite(form_path, form)
        
        # Save annotations
        with open(os.path.join(output_dir, f"syn_form_{i}.txt"), 'w') as f:
            for ann in field_annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
        
        if i % 50 == 0:
            print(f"Generated {i} synthetic forms")


def train_and_evaluate(dataset_dir, model_save_path):
    """
    Train and evaluate the KYC form field detector
    
    Args:
        dataset_dir: Directory containing processed dataset
        model_save_path: Path to save the trained model
    """
    # Initialize the pipeline
    kyc_pipeline = KYCFormTextDetection(model_path=model_save_path)
    
    # Train the model
    history = kyc_pipeline.train_form_detector(os.path.join(dataset_dir, 'train'), epochs=20)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print(f"Model trained and saved to {model_save_path}")
    print(f"Training history saved to training_history.png")


def test_on_kyc_forms(model_path, test_images_dir):
    """
    Test the trained model on real KYC forms
    
    Args:
        model_path: Path to the trained model
        test_images_dir: Directory containing test KYC form images
    """
    # Initialize the pipeline
    kyc_pipeline = KYCFormTextDetection(
        tesseract_path='/usr/share/tesseract',  # Update with your path
        model_path=model_path
    )
    
    # Process test images
    os.makedirs('results', exist_ok=True)
    
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in test_images:
        img_path = os.path.join(test_images_dir, img_file)
        
        # Process KYC form
        result = kyc_pipeline.process_kyc_form(img_path)
        
        # Visualize results
        image = cv2.imread(img_path)
        
        # Draw detected form fields
        for field_id, text in result['raw_fields'].items():
            # Parse coordinates from field_id
            coords = eval(field_id) if field_id.startswith('(') else None
            
            if coords:
                x, y, w, h = coords
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, text[:20], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save visualized result
        output_path = os.path.join('results', f"result_{img_file}")
        cv2.imwrite(output_path, image)
        
        # Save extracted data
        with open(os.path.join('results', f"data_{os.path.splitext(img_file)[0]}.txt"), 'w') as f:
            f.write("RAW FIELDS:\n")
            for field_id, text in result['raw_fields'].items():
                f.write(f"{field_id}: {text}\n")
            
            f.write("\nIDENTIFIED KYC FIELDS:\n")
            for field, value in result['identified_fields'].items():
                f.write(f"{field.upper()}: {value}\n")
        
        print(f"Processed {img_file}, results saved")


if __name__ == "__main__":
    # Step 1: Generate synthetic data
    print("Generating synthetic training data...")
    generate_synthetic_data("synthetic_forms", num_samples=200)
    
    # Step 2: Prepare dataset from synthetic data
    print("Preparing training dataset...")
    prepare_training_dataset("synthetic_forms", "kyc_dataset")
    
    # Step 3: Train model
    print("Training KYC form field detector...")
    train_and_evaluate("kyc_dataset", "kyc_form_detector.h5")
    
    # Step 4: Test on real forms
    print("Testing on KYC forms...")
    test_on_kyc_forms("kyc_form_detector.h5", "test_forms")
    
    print("Pipeline completed!")