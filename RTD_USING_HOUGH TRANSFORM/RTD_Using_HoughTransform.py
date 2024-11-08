import cv2
import numpy as np

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def detect_lines(edges):
    # Use HoughLinesP to detect solid lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    return lines

def draw_lines_and_track_numbers(frame, lines):
    if lines is not None:
        for i in range(len(lines) // 2):
            # Draw lines
            x1, y1, x2, y2 = lines[2 * i][0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x1, y1, x2, y2 = lines[2 * i + 1][0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw track number
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(frame, f'Track {i + 1}', (mid_x - 50, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame width, height, and FPS from the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up the VideoWriter object to save the video with detections
    output_path = video_path.replace('.mkv', '_output.mkv')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        edges = preprocess_frame(frame)
        lines = detect_lines(edges)
        
        draw_lines_and_track_numbers(frame, lines)
        
        # Write the processed frame to the output video
        out.write(frame)
        
        # Show the frame
        cv2.imshow('Rail Track Detection', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved at {output_path}")

if __name__ == "__main__":
    video_path = r'D:\CODE\Projects\Company_Project1\Videos\R_1.mkv'
    main(video_path)
