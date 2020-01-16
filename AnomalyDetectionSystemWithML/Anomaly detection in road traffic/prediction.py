#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.26
#  in conjunction with Tcl version 8.6
#    Dec 30, 2019 12:15:19 PM +04  platform: Windows NT

import sys
import os
from GUI.Events.dataextractionevents import extract_data
from threading import Thread

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

from GUI.Design import prediction_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    prediction_support.set_Tk_var()
    top = clsPredict (root)
    prediction_support.init(root, top)
    root.mainloop()

w = None
def create_clsPredict(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    prediction_support.set_Tk_var()
    top = clsPredict (w)
    prediction_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_clsPredict():
    global w
    w.destroy()
    w = None

class clsPredict:
    def uploadVideo(self):
        videoName = extract_data.uploadVideoFile()
        self.lblVid.insert('1',videoName)
        self.video_name = videoName

    def uploadConfigFile(self):
        config = extract_data.uploadConfigFile()
        self.lblConfig.insert('1',config)
        self.config_name = config

    def uploadModelFile(self):
        model = extract_data.uploadModelFile()
        self.lblmodel.insert('1',model)
        self.model_name = model
    
    def executeScript(self):
        predict_Feature = self.sltFeature.get()
        if predict_Feature == '1. Speed Anomaly Prediction':
            os.system('speedAnomalyTest.py --conf config/'+self.config_name+' --csv CSVFiles/test.csv --video videos/'+self.video_name+' --model Models/'+self.model_name)
        if predict_Feature == '2. Stopping in yellow box Anomaly Prediction':
            os.system('yellow_box_ssd.py --conf config/'+self.config_name+' --csv CSVFiles/test.csv --video videos/'+self.video_name+' --model Models/'+self.model_name)

    def btnPredict_clicked(self):
        self.run_thread('test',self.executeScript)

    def run_thread(self,name,func):
        Thread(target=self.startPrediction,args=(name,func)).start()

    def startPrediction(self,name,func):
        self.TProgressbar1.start(interval=10)
        func()
        self.TProgressbar1.stop()

    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        self.video_name = tk.StringVar()
        self.model_name = tk.StringVar()
        self.config_name = tk.StringVar()

        top.geometry("401x357+1676+147")
        top.minsize(120, 1)
        top.maxsize(2736, 749)
        top.resizable(1, 1)
        top.title("Prediction")
        top.configure(background="#d9d9d9")

        self.Canvas1 = tk.Canvas(top)
        self.Canvas1.place(relx=0.0, rely=0.0, relheight=1.0, relwidth=1.0)
        self.Canvas1.configure(background="#d9d9d9")
        self.Canvas1.configure(borderwidth="2")
        self.Canvas1.configure(insertbackground="black")
        self.Canvas1.configure(relief="ridge")
        self.Canvas1.configure(selectbackground="#c4c4c4")
        self.Canvas1.configure(selectforeground="black")

        self.btnUploadModel = ttk.Button(self.Canvas1,command=self.uploadModelFile)
        self.btnUploadModel.place(relx=0.1, rely=0.42, height=35, width=116)
        self.btnUploadModel.configure(takefocus="")
        self.btnUploadModel.configure(text='''Upload Model''')

        self.btnUploadVid = ttk.Button(self.Canvas1,command=self.uploadVideo)
        self.btnUploadVid.place(relx=0.1, rely=0.14, height=35, width=111)
        self.btnUploadVid.configure(takefocus="")
        self.btnUploadVid.configure(text='''Upload Video''')

        self.btnUploadCnfig = ttk.Button(self.Canvas1,command=self.uploadConfigFile)
        self.btnUploadCnfig.place(relx=0.1, rely=0.56, height=35, width=116)
        self.btnUploadCnfig.configure(takefocus="")
        self.btnUploadCnfig.configure(text='''Upload Config File''')

        self.btnPredict = ttk.Button(self.Canvas1,command = self.btnPredict_clicked)
        self.btnPredict.place(relx=0.125, rely=0.812, height=35, width=96)
        self.btnPredict.configure(takefocus="")
        self.btnPredict.configure(text='''Start Prediction''')

        self.lblVid = ttk.Entry(self.Canvas1)
        self.lblVid.place(relx=0.524, rely=0.14, relheight=0.087, relwidth=0.389)

        self.lblVid.configure(takefocus="")
        self.lblVid.configure(cursor="ibeam")

        self.lblmodel = ttk.Entry(self.Canvas1)
        self.lblmodel.place(relx=0.524, rely=0.42, relheight=0.087
                , relwidth=0.389)
        self.lblmodel.configure(takefocus="")
        self.lblmodel.configure(cursor="ibeam")

        self.lblConfig = ttk.Entry(self.Canvas1)
        self.lblConfig.place(relx=0.524, rely=0.56, relheight=0.087
                , relwidth=0.389)
        self.lblConfig.configure(takefocus="")
        self.lblConfig.configure(cursor="ibeam")

        self.TProgressbar1 = ttk.Progressbar(self.Canvas1,mode='indeterminate')
        self.TProgressbar1.place(relx=0.524, rely=0.812, relwidth=0.399
                , relheight=0.0, height=22)
        self.TProgressbar1.configure(length="160")

        self.TSeparator1 = ttk.Separator(self.Canvas1)
        self.TSeparator1.place(relx=0.05, rely=0.084, relheight=0.644)
        self.TSeparator1.configure(orient="vertical")

        self.TSeparator2 = ttk.Separator(self.Canvas1)
        self.TSeparator2.place(relx=0.05, rely=0.728, relwidth=0.898)

        self.TSeparator3 = ttk.Separator(self.Canvas1)
        self.TSeparator3.place(relx=0.05, rely=0.084, relwidth=0.898)

        self.TSeparator4 = ttk.Separator(self.Canvas1)
        self.TSeparator4.place(relx=0.948, rely=0.084, relheight=0.644)
        self.TSeparator4.configure(orient="vertical")

        self.TLabel1 = ttk.Label(self.Canvas1)
        self.TLabel1.place(relx=0.125, rely=0.308, height=19, width=86)
        self.TLabel1.configure(background="#d9d9d9")
        self.TLabel1.configure(foreground="#000000")
        self.TLabel1.configure(font="TkDefaultFont")
        self.TLabel1.configure(relief="flat")
        self.TLabel1.configure(text='''Choose Feature''')

        self.box_value = tk.StringVar()

        self.sltFeature = ttk.Combobox(self.Canvas1,textvariable=self.box_value,state = 'readonly')
        self.sltFeature.place(relx=0.524, rely=0.28, relheight=0.087
                , relwidth=0.382)
        self.value_list = ['1. Speed Anomaly Prediction','2. Stopping in yellow box Anomaly Prediction']
        self.sltFeature.configure(values=self.value_list)
        self.sltFeature.configure(textvariable=prediction_support.combobox)
        self.sltFeature.configure(takefocus="")

if __name__ == '__main__':
    vp_start_gui()




