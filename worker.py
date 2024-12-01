import xmlrpc.client
import time
import cv2
import torch
import xmlrpc.server
import time
import base64
import numpy as np


# Load YOLOv5 model on CPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
model.conf = 0.4  # Increase confidence threshold

class WorkerClient:
    def __init__(self, master_address):
        self.master = xmlrpc.client.ServerProxy(master_address)
        self.worker_address = f"Worker-{time.time()}"
        self.master.register_worker(self.worker_address)

    def perform_task(self, task):
        encoded_frame = task['encoded_frame']
        port = task['port']

        decoded_bytes = base64.b64decode(encoded_frame)

        # Decode the bytes into an image
        frame = cv2.imdecode(np.frombuffer(decoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inference (detect objects)
        results = model(img_rgb)

        # Results: Draw bounding boxes on the image
        results.render()  # Adds bounding boxes to the image

        # Encode the result image back to Base64
        _, buffer = cv2.imencode('.jpg', img_rgb)
        encoded_result = base64.b64encode(buffer).decode('utf-8')

        return {
            'port': port,
            'encoded_result': encoded_result,  
        }
        # Display the image with detections
        # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        # cv2.imshow(f"Video from Port {port} from worker", img_bgr)  # Display video in a window specific to the port
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return img_bgr
        # return img_bgr


    def start(self):
        while True:
            task_id, task = self.master.request_task(self.worker_address)
            if task_id is not None:
                print(f"{self.worker_address} received task {task_id}")
                result = self.perform_task(task)
                self.master.complete_task(task_id, self.worker_address, result)
            else:
                print(f"{self.worker_address} has no tasks, sleeping...")
                time.sleep(2)  # Sleep before checking again

# Start Worker
def start_worker():
    worker = WorkerClient("http://localhost:8000")
    worker.start()

if __name__ == "__main__":
    start_worker()
