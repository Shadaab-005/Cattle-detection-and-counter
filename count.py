import onnxruntime as rt
import numpy as np
import cv2 


def get_image(image_path):
    image = cv2.imread(image_path)
    
    return image

def preprocess(image,count):
    input_size = (640, 640)  # Replace with your model's input size
    image_resized = cv2.resize(image, input_size)
    image_resized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_resized, (2, 0, 1))
    input_tensor = np.expand_dims(image_transposed, axis=0)
    return input_tensor



def postprocess(outputs, confidence_threshold=0.15, iou_threshold=0.1):
    # Assuming 'outputs' is a list of numpy arrays containing predictions
    predictions = outputs[0]  # Modify this based on your model's output structure
    # Perform Non-Maximum Suppression (NMS) and filter out low confidence boxes
    boxes, scores, class_ids = [], [], []
    for prediction in predictions:
        if prediction[4] > confidence_threshold:  # Assuming the confidence score is at index 4
            boxes.append(prediction[:4])
            scores.append(prediction[4])
            class_ids.append(np.argmax(prediction[5:]))  # Assuming class scores start at index 5

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
    if len(indices)==0:
        print("yes")
        return []
    return [(boxes[i], scores[i], class_ids[i]) for i in indices.flatten()]

def draw_boxes(image, predictions):
    input_size = (640, 640)  # Replace with your model's input size
    image = cv2.resize(image, input_size)
    
    for (box, score, class_id) in predictions:
        x, y, w, h = box
        x=int(x-w/2)
        y=int(y-h/2)
        w=int(w)
        h=int(h)

        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        # cv2.putText(image, f'Class {class_id}: {score:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        count=len(predictions)
        cv2.putText(image, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(image, f"{count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image


## loading model
onnx_model_path=r"C:\Users\shada\Desktop\python task\best.onnx"
ort_session = rt.InferenceSession(onnx_model_path)



## for image
# img_path=r"C:\Users\07032\Downloads\WhatsApp Image 2024-06-05 at 02.03.36.jpeg"
# input_tensor = preprocess(get_image(img_path))
# outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
# predictions = postprocess(outputs[0])
# draw_boxes(get_image(img_path), predictions)

# for video

video_path=r"C:\Users\shada\Desktop\python task\count _cattles.mp4"
cap = cv2.VideoCapture(video_path)

count=0
while cap.isOpened():
    ret,frame=cap.read()
    count+=1
    if count % 6 != 0:
        continue
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    input_tensor = preprocess(frame,count)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
    predictions = postprocess(outputs[0])
    frame=draw_boxes(frame, predictions)
    cv2.imshow("img",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print(count)