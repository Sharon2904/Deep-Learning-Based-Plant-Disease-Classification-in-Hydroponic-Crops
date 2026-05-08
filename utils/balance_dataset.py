import os
import random
from PIL import Image, ImageEnhance

def balance_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    
    # 1. Count images in each class
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            class_counts[class_name] = len(images)
            
    print(f"Original counts: {class_counts}")
    
    # 2. Target count (e.g., matching the majority class or at least 200)
    target_count = 200 
    
    print(f"\nArtificially generating images so every class has at least {target_count} images...")
    
    for class_name, count in class_counts.items():
        if count == 0:
            print(f"Skipping {class_name} because it has 0 images.")
            continue
            
        if count < target_count:
            class_path = os.path.join(data_dir, class_name)
            original_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            images_to_generate = target_count - count
            print(f"Generating {images_to_generate} fake images for {class_name}...")
            
            for i in range(images_to_generate):
                # Pick a random original image to base the fake one on
                src_image_name = random.choice(original_images)
                src_image_path = os.path.join(class_path, src_image_name)
                
                try:
                    img = Image.open(src_image_path)
                    
                    # Apply random augmentations to make it slightly different
                    # Rotate
                    img = img.rotate(random.randint(0, 360))
                    
                    # Flip
                    if random.random() > 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if random.random() > 0.5:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        
                    # Brightness
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.7, 1.3))
                    
                    # Save the new artificial image
                    new_filename = f"aug_{i}_{src_image_name}"
                    img.save(os.path.join(class_path, new_filename))
                    
                except Exception as e:
                    print(f"Failed to augment {src_image_name}: {e}")
                    
    print("\nDataset is now heavily balanced! You can re-run training.")

if __name__ == "__main__":
    balance_dataset()
