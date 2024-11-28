import cv2
import streamlit as st
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Streamlit Setup
st.set_page_config(page_title="PPE KIT DETECTION", layout="wide")
st.title("PPE KIT DETECTION")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi"])

# Add custom styles for borders
st.markdown("""
    <style>
    .frame-box {
        border: 5px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to process video and display frames in two bordered windows
def process_video(uploaded_file):
    # Save uploaded file to temp
    temp_video_path = "uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Open the video file
    cap = cv2.VideoCapture(temp_video_path)

    # Create two placeholders for displaying frames
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="frame-box"><h4>Original Video</h4></div>', unsafe_allow_html=True)
        input_frame_placeholder = st.empty()

    with col2:
        st.markdown('<div class="frame-box"><h4>Processed Video</h4></div>', unsafe_allow_html=True)
        output_frame_placeholder = st.empty()

    # Process frames from the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frames are left

        # Display the input frame in the first window
        frame_rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame_placeholder.image(frame_rgb_input, channels="RGB")

        # Run YOLO inference
        results = model(frame)

        # Draw bounding boxes on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = f"{model.names[class_id]} {confidence:.2f}"

                # Draw the rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the processed frame with bounding boxes in the second window
        frame_rgb_output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame_placeholder.image(frame_rgb_output, channels="RGB")

    cap.release()

# Main Logic
if uploaded_file:
    st.sidebar.text("Processing video... please wait.")
    process_video(uploaded_file)
else:
    st.sidebar.text("Upload a video file to begin processing.")
