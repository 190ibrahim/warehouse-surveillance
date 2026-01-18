
import cv2
import time
from ultralytics import YOLO

def main():
	# setup camera
	cam = cv2.VideoCapture(0)
	if not cam.isOpened():
		print("no camera")
		return
	
	# load YOLO model
	model = YOLO('yolo/yolov8s.pt')
	prev_time = time.time()
	
	print("press q to quit")
	
	while True:
		ret, frame = cam.read()
		if not ret:
			break
		
		# track people in frame
		results = model.track(frame, classes=[0], conf=0.5, tracker="bytetrack.yaml", verbose=False)  # Only person class
		
		tracks_count = 0
		for result in results:
			boxes = result.boxes
			if boxes is not None and boxes.id is not None:
				for bbox, track_id, conf in zip(boxes.xyxy, boxes.id, boxes.conf):
					# get box coordinates and ID
					x1, y1, x2, y2 = map(int, bbox.cpu().numpy())
					track_id = int(track_id.cpu().numpy())
					conf = float(conf.cpu().numpy())
					tracks_count += 1
					
					# draw box and ID
					cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
					cv2.putText(frame, f'ID: {track_id} ({conf:.2f})', (x1, y1-10),
					           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		
		# calculate FPS
		fps = 1 / (time.time() - prev_time)
		prev_time = time.time()
		
		# dhow info
		cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame, f'Tracked: {tracks_count} persons', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		
		cv2.imshow('Tracking', frame)
		
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q') or key == 27:  # 'q' or ESC key
			break
	
	# cleanup
	cam.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()