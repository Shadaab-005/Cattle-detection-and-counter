# **Cattle Detection and Counting Using YOLO and Drone Imagery**

This project aims to detect and count cattle using drone-captured images, helping to survey and evacuate cattle from critical mining areas. The system uses the YOLOv5 model for real-time detection and counting, ensuring safety and operational efficiency in mining zones.

---

## **Table of Contents**

- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Results](#results)  
- [Challenges and Improvements](#challenges-and-improvements)  
- [License](#license)

---

## **Dataset**

- **Name:** WAID (Wildlife Aerial Images from Drone)  
- **Source:** [Mou C, Liu T, Zhu C, Cui X. Waid: A large-scale dataset for wildlife detection with drones. Applied Sciences, 2023.](https://github.com/)  
- **Details:**  
  - **Images:** 14,366 drone-captured images (640x640 pixels).  
  - **Categories:** Sheep, cattle, seals, camels, kiangs, zebras.  
  - **Annotations:** YOLO format bounding boxes.  
  - **Purpose:** Count cattle in mining areas for safe evacuation.

---

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/cattle-detection-yolo.git
   cd cattle-detection-yolo
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Download the WAID dataset and place it in the data/ directory.**

  ## **Usage**
    
    ### **1. Training the YOLOv5 Model:**
       ```bash
       python scripts/train.py --data data.yaml --cfg yolov5s.yaml --epochs 15 --batch-size 8 --name cattle_model
    

### **2\. Exporting the Model to ONNX:**

    python scripts/export.py --weights runs/train/cattle_model/weights/best.pt --include onnx --simplify
    

### **3\. Running Inference on Images or Videos:**

    python scripts/detect.py --weights best.onnx --source data/test_images/ --conf 0.25
    

* * *

**Project Structure**
---------------------

    ├── data/                     # Dataset and annotations
    ├── scripts/                  # Training, export, and detection scripts
    │   ├── train.py              # YOLOv5 training script
    │   ├── export.py             # Export model to ONNX
    │   ├── detect.py             # Detection and counting script
    ├── results/                  # Outputs and logs
    ├── requirements.txt          # Required Python packages
    └── README.md                 # Project documentation
    

* * *

## **Results**

- **Performance Metrics:**
  - Mean Average Precision (mAP): 93.6%
  - Real-time processing capability with optimized inference.

- **Demo Video:**
  To see the model in action, watch the demo video below:
  
  ![Demo Video](https://drive.google.com/file/d/1rvx_wrIUbNTv-hkM5LNFhMWUXrXOE_-q/view?usp=drive_link)


* * *

**Challenges and Improvements**
-------------------------------

### **Challenges:**

*   Handling occlusions where cattle overlap in images.
*   Maintaining accuracy in varying lighting conditions.

### **Future Improvements:**

*   Deploying on edge devices like NVIDIA Jetson Nano for on-site processing.
*   Developing a real-time monitoring dashboard for analytics.



