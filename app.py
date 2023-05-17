import streamlit as st
import cv2
import mediapipe as mp

mp_objectron = mp.solutions.objectron
model = mp_objectron.Objectron(static_image_mode=False)


def process_frame(frame):
    # Convert the BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the results
    results = model.process(frame_rgb)

    # Draw the detected objects on the frame
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_objectron.draw_landmarks(frame, detected_object)

    return frame


def main():
    # Set app title
    st.title("Object Detection with MediaPipe")

    # Create video capture object
    # You can also specify a video file here
    video_capture = cv2.VideoCapture(0)

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()

    # Start the video stream
    while video_capture.isOpened():
        # Read the current frame
        ret, frame = video_capture.read()

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the frame with object detection
        frame_placeholder.image(processed_frame, channels="BGR")

        # Check if 'q' key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    video_capture.release()


if __name__ == '__main__':
    main()
