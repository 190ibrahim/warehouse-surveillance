

import cv2
import time
from ultralytics import YOLO
from src.person import Person

def main():
	cam = cv2.VideoCapture(0)
	
	# Load YOLOv8 model
	model = YOLO('yolov8s.pt')


if __name__ == "__main__":
	main()