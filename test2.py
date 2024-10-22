
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

model.eval()

def detect_objects(image_path):
    img = cv2.imread(image_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    results = model(img) 
    return results

def visualize_results(results, img_path):
    img = cv2.imread(img_path)
    for box in results.xyxy[0]:  
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()  
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, f'Class: {int(cls)} Conf: {conf:.2f}', (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow('Detected Objects', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = '/Users/ouyangyuxuan/Desktop/SUAS-2025/images/odetobeachbub-2048px-umbrella2.webp'  
results = detect_objects(image_path)
visualize_results(results, image_path)
