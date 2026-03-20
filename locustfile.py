from locust import HttpUser, task, between
import os
import random

class WastePredictUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_waste(self):
        # In a real scenario, you'd have a directory of test images
        # We'll simulate by sending a small dummy byte stream if no images exist
        # or pick a random image from data/test if it exists
        test_data_dir = "data/test"
        
        image_to_send = None
        if os.path.exists(test_data_dir):
            # Pick a random image from a subfolder
            all_images = []
            for root, dirs, files in os.walk(test_data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_images.append(os.path.join(root, file))
            
            if all_images:
                image_to_send = random.choice(all_images)

        if image_to_send:
            with open(image_to_send, "rb") as image:
                self.client.post("/predict", files={"file": image})
        else:
            # Fallback for demo purposes if no images found
            # Create a dummy image-like byte stream
            dummy_image = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdcG\xe1\x04\x00\x00\x00\x00IEND\xaeB`\x82"
            self.client.post("/predict", files={"file": ("dummy.png", dummy_image, "image/png")})

    @task
    def check_health(self):
        self.client.get("/health")
