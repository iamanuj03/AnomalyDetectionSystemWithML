from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os

class extract_data:
    def uploadVideoFile():
        name = askopenfilename(initialdir="C:/Users/Anuj/Desktop/opencv-speed-detector/Anomaly detection in road traffic/videos",
                               filetypes =(("Video File", "*.mp4"),("All Files","*.*")),
                               title = "Choose a file."
                               )
        root_ext = os.path.basename(name)
        return root_ext 

    def uploadConfigFile():
        name = askopenfilename(initialdir="C:/Users/Anuj/Desktop/opencv-speed-detector/Anomaly detection in road traffic/config",
                               filetypes =(("config", "*.json"),("All Files","*.*")),
                               title = "Choose a file."
                               )
        root_ext = os.path.basename(name)
        return root_ext 

    def uploadCSVFile():
        name = askopenfilename(initialdir="C:/Users/Anuj/Desktop/opencv-speed-detector/Anomaly detection in road traffic/CSVFiles",
                               filetypes =(("CSVFile", "*.csv"),("All Files","*.*")),
                               title = "Choose a file."
                               )
        root_ext = os.path.basename(name)
        return root_ext 

    def uploadModelFile():
        name = askopenfilename(initialdir="C:/Users/Anuj/Desktop/opencv-speed-detector/Anomaly detection in road traffic/Models",
                               filetypes =(("Model File", "*.model"),("All Files","*.*")),
                               title = "Choose a file."
                               )
        root_ext = os.path.basename(name)
        return root_ext 
    




