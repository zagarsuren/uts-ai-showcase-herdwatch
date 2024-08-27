import pickle
from django.conf import settings
#from cam_app import views
from django.http import StreamingHttpResponse
import sqlite3
import datetime

# import some common libraries
import numpy as np
import os, json, cv2, random, glob, uuid
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from pathlib import Path
import time
# import torch
weights_dir = settings.YOLOV8_WEIGTHS_DIR
yolov8m_model = YOLO(os.path.join(weights_dir, "final_model.pt"))

yolov8m_model.export(format='openvino')