# UI.py
# Handles the user interface (presumably with tkinter or CLI menu) for the video simulation software.
# Functions should manage launching the GUI, handling user input, and triggering runs.

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import os
from core.parameters import Parameters
from tkinter import messagebox
from PIL import Image, ImageTk
import Pmw
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import core.main as main
import threading
import metrics.ploting as ploting




########################## Main View #####################################

# Video player

class VideoPlayer:

    def __init__(self, root, width_entry, height_entry, fps_entry):
        self.root = root
        self._video_source = ""
        self._width = 144 # Default
        self._height = 144 # Default
        self._fps = 2  # Default 
        self.create_widgets()
        self.cap = None
        self.is_playing = False  # Track whether the video is currently playing
        self.width_entry = width_entry
        self.height_entry = height_entry
        self.fps_entry = fps_entry


        self.fps_var = tk.StringVar()
        self.fps_var.set(self._fps)
        self.fps_var.trace("w", lambda *args: self.on_value_change("fps", self.fps_var))
        self.width_var = tk.StringVar()
        self.width_var.set(self._width)
        self.width_var.trace("w", lambda *args: self.on_value_change("width", self.width_var))
        self.height_var = tk.StringVar()
        self.height_var.set(self._height)
        self.height_var.trace("w", lambda *args: self.on_value_change("height", self.height_var))
        self.fps_entry.config(textvariable=self.fps_var)
        self.width_entry.config(textvariable=self.width_var)
        self.height_entry.config(textvariable=self.height_var)

    def reload(self):
        self.cap = cv2.VideoCapture(self._video_source)
        if self.cap.isOpened():
            self.show_first_frame()
            self.update_canvas_size()

    # Function to track the value changing on input entries
    def on_value_change(self,var_name, string_var):
        # Modifiying the canvas that holds the video preview and make the appropriate changes based on the new variables
        # Get the old parameters if one value changed reload the player
        if  len(string_var.get()) !=  0:
            try:
                if(var_name == "fps"):
                    new_fps = float(string_var.get())
                    assert new_fps > 0, f"Parameter: {var_name} must be positive."
                    if(self._fps != new_fps):
                        self._fps = new_fps
                        #print("update player fps changed")
                        self.reload()
                elif(var_name == "width"):
                    new_width = int(string_var.get())
                    assert new_width > 0, f"Parameter: {var_name} must be positive."
                    if(self.width != new_width):
                        self.width = new_width
                        #print("update player width changed")
                        self.reload()
                elif(var_name == "height"):
                    new_height= int(string_var.get())
                    assert new_height > 0, f"Parameter: {var_name} must be positive."
                    if(self.height != new_height):
                        #print("update player height changed")
                        self.height = new_height
                        self.reload()
            except ValueError as err:
                if var_name == "fps":
                    messagebox.showerror("Value error", f"Parameter: {var_name} must be of a numerical value.")
                else:
                    messagebox.showerror("Value error", f"Parameter: {var_name} must be of an integer value.")
            except AssertionError as ass_err:
                messagebox.showerror("Value error", f" {err_}.")

            #print(f"Variable {var_name} changed to {string_var.get()}")

    @property
    def video_source(self):
        return self._video_source

    @video_source.setter
    def video_source(self, path):
        self._video_source = path
        self.cap = cv2.VideoCapture(self._video_source)
        if self.cap.isOpened():
            self._fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # Get FPS from video file using int() to floor the fps to avoid getting 0 in core.main_frame for loop line 266
            self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get width from video file
            self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get height from video file
            self.width_entry.delete(0, tk.END)  # Delete the old value
            self.width_entry.insert(0, self.width) 
            self.height_entry.delete(0, tk.END)  # Delete the old value
            self.height_entry.insert(0, self.height)
            self.fps_entry.delete(0, tk.END)  # Delete the old value
            self.fps_entry.insert(0, self.fps)
            self.show_first_frame()
            self.update_canvas_size()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self.update_canvas_size()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self.update_canvas_size()

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value

    def create_widgets(self):
        tk.Label(self.root, text="Captured video preview:").grid(row=0, column=0, sticky=tk.W)

        self.canvas = tk.Canvas(self.root, width=self._width, height=self._height)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.grid(row=2, column=0, sticky="ew")

        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)
        self.controls_frame.grid_columnconfigure(2, weight=1)

        self.play_icon = ImageTk.PhotoImage(Image.open(r"src\res\play.png"))
        self.btn_play = tk.Button(self.controls_frame, image=self.play_icon, width=30, height=30, command=self.play_video)
        self.btn_play.grid(row=0, column=0, padx=10, pady=10)

        self.pause_icon = ImageTk.PhotoImage(Image.open(r"src\res\pause.png"))
        self.btn_pause = tk.Button(self.controls_frame, image=self.pause_icon, width=30, height=30, command=self.pause_video)
        self.btn_pause.grid(row=0, column=1, padx=10, pady=10)

        self.stop_icon = ImageTk.PhotoImage(Image.open(r"src\res\stop.png"))
        self.btn_stop = tk.Button(self.controls_frame, image=self.stop_icon, width=30, height=30, command=self.stop_video)
        self.btn_stop.grid(row=0, column=2, padx=10, pady=10)

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def update_canvas_size(self):
        self.canvas.config(width=self._width, height=self._height)

    def play_video(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self._video_source)
        if self.cap.isOpened():
            self.is_playing = True
            self.update_frame()

    def pause_video(self):
        if self.cap:
            self.is_playing = False

    def stop_video(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        self.canvas.delete("all")
        self.cap = None

    def update_frame(self):
        if self.cap and self.cap.isOpened() and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self._width, self._height))
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.canvas.image = frame

                frame_delay = int(1000 / self._fps)  # Calculate frame delay based on FPS
                self.root.after(frame_delay, self.update_frame)  # Update frame based on FPS
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                self.update_frame()


    def show_first_frame(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Set to the first frame
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self._width, self._height))
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.canvas.image = frame


# Helper function to create labels and entries
def create_label_entry(root, text, row, default_value="", entry_state='normal',unit = ''):
        tk.Label(root, text=text).grid(row=row, column=0, sticky=tk.W)
        entry = tk.Entry(root)
        entry.grid(row=row, column=1)
        entry.insert(0, default_value)
        entry.config(state=entry_state)
        tk.Label(root, text=unit).grid(row=row, column=2, sticky=tk.W)
        return entry

# Helper function to create labels and comboboxes
def create_label_combobox(root, text, row, default_value, values):
    tk.Label(root, text=text).grid(row=row, column=0, sticky=tk.W)
    combobox = ttk.Combobox(root, values=values, width=17)
    combobox.grid(row=row, column=1)
    combobox.set(default_value)  # Ensure the combobox shows the default value
    return combobox

def check_video_file_exists(file_path):
    if  not os.path.exists(file_path) and not os.path.isfile(file_path):
        messagebox.showerror("File Check", "Video file does not exist.")
        return False
    else:
        return True
        

def check_and_create_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            #print(f"Directory '{directory_path}' created.")
        else:
            #print(f"Directory '{directory_path}' already exists.")
            return True
    except Exception as err:
            messagebox.showerror("Directory Check", f"Directory error{err}.")
            return False

    

def open_device_params(root):

    def set_device_params():    
        try:
            Parameters.POWER = float(power_entry.get())
            Parameters.FREQUENCY = float(frequency_entry.get()) 
            Parameters.PROC_CLOCK = float(proc_clock_entry.get())
            Parameters.CAPTURE_E_PER_BLOCK = float(capture_per_block_entry.get())
            Parameters.CYCLES_PER_DIV = int(cycles_div_entry.get()) 
            Parameters.CYCLES_PER_ADD = int(cycles_add_entry.get()) 
            Parameters.CYCLES_PER_FADD = int(cycles_fadd_entry.get()) 
            Parameters.CYCLES_PER_SHIFT = int(cycles_shift_entry.get()) 
            Parameters.CYCLES_PER_MUL = int(cycles_mul_entry.get()) 
            Parameters.CYCLES_PER_FMUL = int(cycles_fmul_entry.get())
            Parameters.CLA_ADD_NB = int(cycles_add_nb_entry.get())
            Parameters.CLA_MUL_NB = int(cycles_mul_nb_entry.get()) 
            assert (Parameters.POWER >0 and Parameters.FREQUENCY >0 and Parameters.PROC_CLOCK >0 and Parameters.CAPTURE_E_PER_BLOCK >0 and 
            Parameters.CYCLES_PER_DIV >0 and Parameters.CYCLES_PER_ADD >0 and Parameters.CYCLES_PER_FADD >0 and  Parameters.CYCLES_PER_SHIFT >0 and 
            Parameters.CYCLES_PER_MUL >0 and Parameters.CYCLES_PER_FMUL >0 and Parameters.CLA_ADD_NB  >0 and Parameters.CLA_MUL_NB >0), "All values must be non-negative"
        except ValueError as err:
            messagebox.showerror("Invalid Input", f"Invalid value. Expected a ({err}).")
            return
        except AssertionError as err:
            messagebox.showerror("Invalid Input", f"Invalid value. Expected a non-negative number ({err}).")
            return
        messagebox.showinfo("Saved!","Device Parameters setted.")
        #print("Device params setted!!")
        top_wind.destroy()

            
    top_wind = tk.Toplevel(root)
    top_wind.attributes('-topmost', True)
    top_wind.resizable(False, False)
    top_wind.grab_set()
    top_wind.title("Processing Unit Parameters")   
    
    power_entry = create_label_entry(top_wind, "POWER:", 0, Parameters.POWER,unit="J")
    frequency_entry = create_label_entry(top_wind, "FREQUENCY:", 1, Parameters.FREQUENCY,unit="Hz")
    proc_clock_entry = create_label_entry(top_wind, "PROC_CLOCK:", 2, Parameters.PROC_CLOCK,unit="")
    capture_per_block_entry = create_label_entry(top_wind, "CAPTURE_E_PER_BLOCK:", 3, Parameters.CAPTURE_E_PER_BLOCK,unit="J")
    cycles_div_entry = create_label_entry(top_wind, "CYCLES_PER_DIV:", 4, Parameters.CYCLES_PER_DIV,unit="Cycle")
    cycles_add_entry = create_label_entry(top_wind, "CYCLES_PER_ADD:", 5, Parameters.CYCLES_PER_ADD,unit="Cycle")
    cycles_fadd_entry = create_label_entry(top_wind, "CYCLES_PER_FADD:", 6, Parameters.CYCLES_PER_FADD,unit="Cycle")
    cycles_shift_entry = create_label_entry(top_wind, "CYCLES_PER_SHIFT:", 7, Parameters.CYCLES_PER_SHIFT,unit="Cycle")
    cycles_mul_entry = create_label_entry(top_wind, "CYCLES_PER_MUL:", 8, Parameters.CYCLES_PER_MUL,unit="Cycle")
    cycles_fmul_entry = create_label_entry(top_wind, "CYCLES_PER_FMUL:", 9,  Parameters.CYCLES_PER_FMUL,unit="Cycle")
    cycles_add_nb_entry = create_label_entry(top_wind, "CLA_ADD_NB:", 10, Parameters.CLA_ADD_NB,unit="Cycle")
    cycles_mul_nb_entry = create_label_entry(top_wind, "CLA_MUL_NB:", 11, Parameters.CLA_MUL_NB,unit="Cycle")

    # Adding the button with proper padding
    set_button = tk.Button(top_wind, text="Set Parameters", command=set_device_params)
    set_button.grid(row=12, column=0, columnspan=2, padx=5, pady=10)
    





def open_env_parameters(root):


    def set_env_params():
        try:
            Parameters.DISTANCE = float(distance_entry.get())
            Parameters.HUMIDITY_LEVEL = float(humidity_entry.get())
            Parameters.VEGETATION_DENSITY_LEVEL = float(vegetation_entry.get())
            Parameters.scale_factor = float(scale_entry.get())
            Parameters.ENVIRONMENT = environment_var.get()
            assert (Parameters.DISTANCE > 0 and Parameters.HUMIDITY_LEVEL > 0 and
            Parameters.VEGETATION_DENSITY_LEVEL > 0 and Parameters.scale_factor > 0)
        except ValueError as err:
            messagebox.showerror("Invalid Input", f"Invalid value. Expected a float ({err}).")
            return
        except AssertionError as err:
            messagebox.showerror("Invalid Input", f"Invalid value. Expected a non-negative number ({err}).")
            return
        messagebox.showinfo("Saved!","Environment Parameters Setted.")
        #print("Environment params setted!!")
        top_window.destroy()

    top_window = tk.Toplevel(root)
    top_window.title("Top Level Parameters")
    top_window.attributes('-topmost', True)
    top_window.resizable(False, False)
    top_window.grab_set()

    # Environment Parameters
    distance_entry = create_label_entry(top_window, "Distance:", 0, Parameters.DISTANCE)
    humidity_entry = create_label_entry(top_window, "Humidity Level:", 1, Parameters.HUMIDITY_LEVEL)
    vegetation_entry = create_label_entry(top_window, "Vegetation Density Level:", 2, Parameters.VEGETATION_DENSITY_LEVEL)
    scale_entry = create_label_entry(top_window, "Scale Factor:", 3, Parameters.scale_factor)
    environment_var = create_label_combobox(top_window,"Environment:",4,"Default",["Default","wetland"])
    set_button = tk.Button(top_window, text="Set Parameters", command=set_env_params)
    set_button.grid(row=5, column=0, columnspan=2, padx=5, pady=10)









def init_root_view():
    def browse_video_path():
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.y4m")])
        if video_path:
            video_path_entry.delete(0, tk.END)
            video_path_entry.insert(0, video_path)
            player.video_source = video_path

    def browse_directory_path():
        directory_path = filedialog.askdirectory()
        if directory_path:
            output_directory_entry.delete(0, tk.END)
            output_directory_entry.insert(0, directory_path)

    # Function to retreive the parameters
    def set_parameters():
        show_progress_bar()
        try:
            Parameters.method = coding_method.get()
            assert Parameters.method != '', "Must Select a coding method" 

            Parameters.captured_video_path = video_path_entry.get()
            assert Parameters.captured_video_path != "", "Must Select a Video File"

            if not check_video_file_exists(Parameters.captured_video_path): return

            Parameters.output_directory = output_directory_entry.get()
            if Parameters.output_directory != '' : 
                if not check_and_create_directory(Parameters.output_directory):
                    return


            Parameters.fps = int(fps_entry.get())
            assert Parameters.fps > 0, "FPS must be positive"

            Parameters.default_width = int(width_entry.get())
            Parameters.default_height = int(height_entry.get())
            assert (Parameters.default_width > 0 and Parameters.default_width % 8 == 0) and (Parameters.default_height > 0 and Parameters.default_height % 8 == 0), "Width and height must be positive and divisible by 8"

            Parameters.gop_coefficient = int(gop_entry.get())
            assert Parameters.gop_coefficient > 0, "GOP must be positive"

            Parameters.quality_factor = int(qf_entry.get()) if coding_method.get() == 'Non-ROI' else 0
            assert 0 < Parameters.quality_factor < 100 if coding_method.get() == 'Non-ROI' else True, "Quality factor must be >0 and <100"

            Parameters.high_quality_factor = int(hqf_entry.get()) if coding_method.get() == 'ROI' else 0
            assert 0 < Parameters.high_quality_factor < 100 if coding_method.get() == 'ROI' else True, "High quality factor must be >0 and <100"

            Parameters.low_quality_factor = int(lqf_entry.get()) if coding_method.get() == 'ROI' else 0
            assert 0 < Parameters.low_quality_factor < 100 if coding_method.get() == 'ROI' else True, "Low quality factor must be >0 and <100"

            Parameters.zone_size = int(zone_size_entry.get())
            assert (Parameters.zone_size > 0 and Parameters.zone_size % 8 == 0) , "Zone size must be >0 and divisible by 8"
    
            
            Parameters.DCT = dct_var.get()

            Parameters.entropy_coding = entropy_var.get()

            Parameters.level_numbers = int(level_entry.get())
            assert Parameters.level_numbers > 0, "Level numbers must be positive"

            Parameters.packed_load_size = int(packed_load_size_entry.get())
            assert Parameters.packed_load_size > 0, "Packed load size must be positive"

            Parameters.interleaving = interleaving_var.get()

            Parameters.threshold = int(threshold_entry.get())
            assert Parameters.threshold >= 0, "Threshold must be non-negative"

            Parameters.max_level_S = int(max_level_S_entry.get())
            assert Parameters.max_level_S >= 0, f"Max level S must be positive or == 0 var = {Parameters.max_level_S}"

            Parameters.w1 = int(w1_entry.get())
            assert Parameters.w1 >= 0, "w1 must be non-negative"

            Parameters.w2 = int(w2_entry.get())
            assert Parameters.w2 >= 0, "w2 must be non-negative"

            Parameters.w3 = int(w3_entry.get())
            assert Parameters.w3 >= 0, "w3 must be non-negative"

            Parameters.roi_threshold = int(roi_threshold_entry.get())
            assert Parameters.roi_threshold >= 0, "ROI threshold must be non-negative"

        except ValueError as err:
            messagebox.showerror("Invalid Input", f"Invalid value. Expected a {err}.")
            return
        except AssertionError as err:
            messagebox.showerror("Invalid Input", f"{err}.")
            return

        threading.Thread(target=run_coding_thread).start()

        
    
    
            


    #Helper function changing coding method
    def toggle_parameters():
        if coding_method.get() == 'ROI':
            hqf_entry.config(state='normal')
            lqf_entry.config(state='normal')
            w1_entry.config(state='normal')
            w2_entry.config(state='normal')
            w3_entry.config(state='normal')
            roi_threshold_entry.config(state='normal')
            qf_entry.config(state='disabled')
            
        else:
            hqf_entry.config(state='disabled')
            lqf_entry.config(state='disabled')
            w1_entry.config(state='disabled')
            w2_entry.config(state='disabled')
            w3_entry.config(state='disabled')
            roi_threshold_entry.config(state='disabled')
            qf_entry.config(state='normal')

    # Function to restore all the parameters to default 
    def default_parameters():
        #Reset the parameters
        Parameters.reset()
        # Parameters are reset to default and change the entries on the screen
        fps_entry.delete(0, tk.END)  # Clear the current text
        fps_entry.insert(0, Parameters.fps) 

        width_entry.delete(0, tk.END)
        width_entry.insert(0, Parameters.default_width) 

        height_entry.delete(0, tk.END)
        height_entry.insert(0, Parameters.default_height)

        gop_entry.delete(0, tk.END)
        gop_entry.insert(0, Parameters.gop_coefficient)

        qf_entry.delete(0, tk.END)
        qf_entry.insert(0, Parameters.quality_factor)
        
        hqf_entry.delete(0, tk.END)
        hqf_entry.insert(0, Parameters.high_quality_factor)
        
        lqf_entry.delete(0, tk.END)
        lqf_entry.insert(0, Parameters.low_quality_factor)
        
        zone_size_entry.delete(0, tk.END)
        zone_size_entry.insert(0, Parameters.zone_size)

        dct_var.set(Parameters.DCT) 
        
        entropy_var.set(Parameters.entropy_coding)
        
        level_entry.delete(0, tk.END)
        level_entry.insert(0, Parameters.level_numbers)
        
        packed_load_size_entry.delete(0, tk.END)
        packed_load_size_entry.insert(0, Parameters.packed_load_size) 
        
        interleaving_var.set("None")

        threshold_entry.delete(0, tk.END)
        threshold_entry.insert(0, Parameters.threshold)
                
        max_level_S_entry.delete(0, tk.END)
        max_level_S_entry.insert(0, Parameters.max_level_S) 
        
        w1_entry.delete(0, tk.END)
        w1_entry.insert(0, Parameters.w1)

        w2_entry.delete(0, tk.END)
        w2_entry.insert(0, Parameters.w2)

        w3_entry.delete(0, tk.END)
        w3_entry.insert(0, Parameters.w3)

        roi_threshold_entry.delete(0, tk.END)
        roi_threshold_entry.insert(0, Parameters.roi_threshold)

        coding_method.set(Parameters.method)

    def show_progress_bar():
        for widget in params_frame.winfo_children():
            widget.configure(state='disabled')
        progress_bar.grid()  # Show the progress bar
        progress_bar.start(10)



    def run_coding_thread():
        # Call main coding function
        Parameters.setup_directories()
        Parameters.print_params()
        main.run_coding()
        main.run_decoding()
        
        # Use the event loop to schedule the opening of the results view
        root.after(0, lambda: open_results_view(root))


    # Main Window
    root = tk.Tk()

    icon = tk.PhotoImage(file=r'src\res\icon.png')
    root.iconphoto(True, icon)
    root.title("Set Parameters")
    tool_tip = Pmw.Balloon(root)
   
   
    # Configure grid weights for main window
    root.grid_columnconfigure(0, weight=0)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_rowconfigure(3, weight=1)  # Added to configure grid for progress bar

    


    # Empty frames to center params_frame vertically
    empty_frame_top = tk.Frame(root)
    empty_frame_top.grid(row=0, column=0,sticky='n')
    # Parameters frame
    params_frame = tk.Frame(root)
    params_frame.grid(row=1, column=0, padx=10, pady=10,sticky='n')

    empty_frame_bottom = tk.Frame(root)
    empty_frame_bottom.grid(row=2, column=0,sticky='n')


    tk.Label(params_frame, text="Captured Video Path:").grid(row=0, column=0, sticky=tk.W)
    video_path_entry = tk.Entry(params_frame, width=50)
    video_path_entry.grid(row=0, column=1)
    browse_button = tk.Button(params_frame, text="Browse", command=browse_video_path)
    browse_button.grid(row=0, column=2)

    tk.Label(params_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W)
    output_directory_entry = tk.Entry(params_frame, width=50)
    output_directory_entry.grid(row=1, column=1)
    browse_directory_button = tk.Button(params_frame, text="Browse", command=browse_directory_path)
    browse_directory_button.grid(row=1, column=2)
    fps_entry = create_label_entry(params_frame, "FPS:", 2, Parameters.fps, unit="Fps")
    width_entry = create_label_entry(params_frame, "Width:", 3, Parameters.default_width, unit="pixel")
    height_entry = create_label_entry(params_frame, "Height:", 4, Parameters.default_height, unit="pixel")
    gop_entry = create_label_entry(params_frame, "GOP Coefficient:", 5, Parameters.gop_coefficient)
    qf_entry = create_label_entry(params_frame, "Quality Factor:", 6, Parameters.quality_factor)
    hqf_entry = create_label_entry(params_frame, "High Quality Factor:", 7, Parameters.high_quality_factor, entry_state='disabled')
    lqf_entry = create_label_entry(params_frame, "Low Quality Factor:", 8, Parameters.low_quality_factor, entry_state='disabled')
    zone_size_entry = create_label_entry(params_frame, "Zone size:", 9, Parameters.zone_size)

    dct_var = create_label_combobox(params_frame, "DCT:", 10, Parameters.DCT, ["CLA", "BiN_DCT", "LLM"])
    entropy_var = create_label_combobox(params_frame, "Entropy Coding:", 11, Parameters.entropy_coding, ["RLE_EG", "RLE", "EG"])
    level_entry = create_label_entry(params_frame, "Level Numbers:", 12, Parameters.level_numbers)
    packed_load_size_entry = create_label_entry(params_frame, "Packed Load Size:", 13, Parameters.packed_load_size, unit="Kbytes")
    interleaving_var = create_label_combobox(params_frame, "Interleaving:", 14, "None", ["None", "torus"])

    threshold_entry = create_label_entry(params_frame, "Threshold:", 15, Parameters.threshold)
    max_level_S_entry = create_label_entry(params_frame, "Max Level S:", 16, Parameters.max_level_S)
    w1_entry = create_label_entry(params_frame, "W1:", 17, Parameters.w1, entry_state='disabled')
    w2_entry = create_label_entry(params_frame, "W2:", 18, Parameters.w2, entry_state='disabled')
    w3_entry = create_label_entry(params_frame, "W3:", 19, Parameters.w3, entry_state='disabled')
    roi_threshold_entry = create_label_entry(params_frame, "ROI Threshold:", 20, Parameters.roi_threshold, entry_state='disabled')

    coding_method = tk.StringVar(value='Non-ROI')
    roi_radiobutton = tk.Radiobutton(params_frame, text="ROI", variable=coding_method, value='ROI', command=toggle_parameters)
    roi_radiobutton.grid(row=21, column=0, sticky=tk.W)
    non_roi_radiobutton = tk.Radiobutton(params_frame, text="Non-ROI", variable=coding_method, value='Non-ROI', command=toggle_parameters)
    non_roi_radiobutton.grid(row=21, column=1, sticky=tk.W)

    top_level_button = tk.Button(params_frame, text="Environment Parameters", command=lambda: open_env_parameters(root))
    top_level_button.grid(row=22, column=0)
    tool_tip.bind(top_level_button, "Open Environment Parameters")


    device_param_button = tk.Button(params_frame,text="Device Parameters",command=lambda: open_device_params(root))
    device_param_button.grid(row=22, column=1)
    tool_tip.bind(device_param_button, "Open Device Parameters")

    dafault_param_button = tk.Button(params_frame,text="Default Parameters",command=default_parameters)
    dafault_param_button.grid(row=22, column=2)
    tool_tip.bind(dafault_param_button, "Reset to default Parameters")

    set_button = tk.Button(params_frame, text="Start Simulation", command=set_parameters)
    set_button.grid(row=23, column=0, columnspan=3)

    # Video player frame
    video_frame = tk.Frame(root)
    video_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
    player = VideoPlayer(video_frame, width_entry, height_entry, fps_entry)
  
    # Progress bar at the bottom of the window
    progress_bar = ttk.Progressbar(root, mode='indeterminate')
    progress_bar.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")
    progress_bar.grid_remove()  # Initially hide the progress bar

    return root

########################## Result View  #####################################

def load_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()



def colorize_text(text_widget, lines):
    for i, line in enumerate(lines):
        text_widget.insert(tk.END, line)
        if 'M' in line:
            text_widget.tag_add(str(i), f"{i+1}.0", f"{i+1}.end")
            text_widget.tag_config(str(i), foreground="red")
        elif 'S' in line:
            text_widget.tag_add(str(i), f"{i+1}.0", f"{i+1}.end")
            text_widget.tag_config(str(i), foreground="blue")
        elif 'R' in line:
            text_widget.tag_add(str(i), f"{i+1}.0", f"{i+1}.end")
            text_widget.tag_config(str(i), foreground="purple")



def init_results_view():
    results_view = tk.Tk()
    results_view.title("Results")
    icon = tk.PhotoImage(file=r'src\res\icon.png')
    results_view.iconphoto(True, icon)
    tool_tip = Pmw.Balloon(results_view)
    notebook = ttk.Notebook(results_view)

    data = get_data()
    rt_frame_data = data["rt_frame"]
    st_frame_data= data["st_frame"]
    st_packet_data = data["st_packet"]

    trace_tab = ttk.Notebook(notebook)
    trace_frame = ttk.Frame(trace_tab)

    create_tab(trace_tab, "st_packets", os.path.join(Parameters.trace_file_path, 'st-packet'))
    create_tab(trace_tab, "st_frames", os.path.join(Parameters.trace_file_path, 'st-frame'))
    create_tab(trace_tab, "rt_frames", os.path.join(Parameters.trace_file_path, 'rt-frame'))

    trace_tab.pack(expand=True, fill='both')
    notebook.add(trace_tab, text="Trace")

    

    frames_frame = ttk.Frame(notebook)
    frames_notebook = ttk.Notebook(frames_frame)

    captured_frames_tab = ttk.Frame(frames_notebook)
    decoded_frames_tab = ttk.Frame(frames_notebook)

    plots_frame = ttk.Frame(notebook)
    plots_notebook = ttk.Notebook(plots_frame)

    frames_notebook.add(captured_frames_tab, text="Captured Frames")
    frames_notebook.add(decoded_frames_tab, text="Decoded Frames")

    load_images(captured_frames_tab, Parameters.captured_frame_dir)
    load_images(decoded_frames_tab, Parameters.decoded_frames_dir)

    frames_notebook.pack(expand=True, fill='both')
    frames_frame.pack(expand=True, fill='both')
    notebook.add(frames_frame, text="Frames")

    #Plots Tabs
    quality_metrics_tab = ttk.Frame(plots_frame)
    rate_distorion_tab = ttk.Frame(plots_frame)
    bitrate_tab = ttk.Frame(plots_frame)
    bpp_tab = ttk.Frame(plots_frame)
    energy_tab = ttk.Frame(plots_frame)
    frame_size_tab = ttk.Frame(plots_frame)
    noise_tab = ttk.Frame(plots_frame)
    packets_size_over_time_tab = ttk.Frame(plots_frame)


    plots_notebook.add(quality_metrics_tab, text="Quality Metrics")
    plots_notebook.add(rate_distorion_tab, text="Quality Metrics Rate Distortion")
    plots_notebook.add(bitrate_tab, text="Bitrate")
    plots_notebook.add(bpp_tab, text="Bits Per Pixel")
    plots_notebook.add(energy_tab, text="Energy")
    plots_notebook.add(frame_size_tab, text="Frames Size")
    plots_notebook.add(noise_tab, text="Generated Noise")
    plots_notebook.add(packets_size_over_time_tab, text="Packets Over Time")
    plots_notebook.pack(expand=True, fill='both')
    plots_frame.pack(expand=True, fill='both')
    notebook.add(plots_frame, text="Plots")




    def init_plot(master_tab):
        fig, ax = plt.subplots(figsize=(8, 6))
        canvas = FigureCanvasTkAgg(fig, master=master_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, master_tab)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return fig, ax, canvas


    ##### fig, ax, toolbar for quality metrics
    fig, ax , canva =  init_plot(quality_metrics_tab)
    ##### fig, ax, toolbar for rate distortion
    fig_rate_distortion, ax_rate_distortion , canva_rate_distortion =  init_plot(rate_distorion_tab)
    ##### fig, ax, toolbar for bitrate
    fig_bit, ax_bit, canvas_bit = init_plot(bitrate_tab)
    ##### fig, ax, toolbar for bpp
    fig_bpp, ax_bpp, canvas_bpp = init_plot(bpp_tab)
    ##### fig, ax, toolbar for energy
    fig_enrgy, ax_enrgy, canvas_enegy = init_plot(energy_tab)
    ##### fig, ax, toolbar for data_size
    fig_size, ax_size, canvas_size = init_plot(frame_size_tab)
    ##### fig, ax, toolbar for generated noise
    fig_noise, ax_noise, canvas_noise = init_plot(noise_tab)
    ##### fig, ax, toolbar for packets over time
    fig_packets, ax_packets, canvas_packets = init_plot(packets_size_over_time_tab)

    ploting.plot_bitrate(ax_bit, st_frame_data)
    ploting.plot_bpp(ax_bpp, st_frame_data)   
    ploting.plot_data_size(ax_size, st_frame_data)
    ploting.plot_packet_size_over_time(ax_packets, st_packet_data)


    show_psnr_var = tk.BooleanVar(value=True)
    show_ssim_var = tk.BooleanVar(value=False)
    show_ref_psnr_var = tk.BooleanVar(value=False)
    show_ref_ssim_var = tk.BooleanVar(value=False)
    show_brisque_var = tk.BooleanVar(value=False)


    def update_quality_metrics_plot():
        if show_psnr_var.get():
            ploting.plot_psnr(ax, rt_frame_data)
        elif show_ssim_var.get():
            ploting.plot_ssim(ax, rt_frame_data)
        elif show_ref_psnr_var.get():
            ploting.plot_ref_psnr(ax,st_frame_data)
        elif show_ref_ssim_var.get():
            ploting.plot_ref_ssim(ax,st_frame_data)
        elif show_brisque_var.get():
            ploting.plot_brisque(ax,st_frame_data,rt_frame_data)

    checkboxes_frame = ttk.Frame(quality_metrics_tab)
    checkboxes_frame.pack(side=tk.TOP, fill=tk.X)

    ttk.Checkbutton(checkboxes_frame, text='Show PSNR', variable=show_psnr_var, command=lambda: [show_ssim_var.set(False), show_ref_psnr_var.set(False),show_ref_ssim_var.set(False),show_brisque_var.set(False), update_quality_metrics_plot()]).pack(side=tk.LEFT)
    ttk.Checkbutton(checkboxes_frame, text='Show SSIM', variable=show_ssim_var, command=lambda: [show_psnr_var.set(False),show_ref_psnr_var.set(False),show_ref_ssim_var.set(False),show_brisque_var.set(False), update_quality_metrics_plot()]).pack(side=tk.LEFT)
    ttk.Checkbutton(checkboxes_frame, text='Show Refrerence PSNR', variable=show_ref_psnr_var, command=lambda: [show_ssim_var.set(False), show_psnr_var.set(False), show_ref_ssim_var.set(False),show_brisque_var.set(False), update_quality_metrics_plot()]).pack(side=tk.LEFT)
    ttk.Checkbutton(checkboxes_frame, text='Show Reference SSIM', variable=show_ref_ssim_var, command=lambda: [show_psnr_var.set(False),show_ssim_var.set(False), show_ref_psnr_var.set(False),show_brisque_var.set(False), update_quality_metrics_plot()]).pack(side=tk.LEFT)
    ttk.Checkbutton(checkboxes_frame, text='Show BRISQUE', variable=show_brisque_var, command=lambda: [show_psnr_var.set(False),show_ssim_var.set(False), show_ref_psnr_var.set(False),show_ref_ssim_var.set(False), update_quality_metrics_plot()]).pack(side=tk.LEFT)

    update_quality_metrics_plot()

    ################################ rate distortion ##################################################
    show_psnr_rate_var = tk.BooleanVar(value=True)
    show_ssim_rate_var = tk.BooleanVar(value=False)
    show_brisque_rate_var = tk.BooleanVar(value=False)


    def update_quality_metrics_rate_plot():
        if show_psnr_rate_var.get():
            ploting.plot_psnr_bpp(ax_rate_distortion, rt_frame_data,st_frame_data)
        elif show_ssim_rate_var.get():
            ploting.plot_ssim_bpp(ax_rate_distortion, rt_frame_data,st_frame_data)
        elif show_brisque_rate_var.get():
            ploting.plot_brisque_bpp(ax_rate_distortion, rt_frame_data,st_frame_data)

    checkboxes_frame_ = ttk.Frame(rate_distorion_tab)
    checkboxes_frame_.pack(side=tk.TOP, fill=tk.X)

    ttk.Checkbutton(checkboxes_frame_, text='Show PSNR Rate Distortion', variable=show_psnr_rate_var, command=lambda: [show_ssim_rate_var.set(False), show_brisque_rate_var.set(False), update_quality_metrics_rate_plot()]).pack(side=tk.LEFT)
    ttk.Checkbutton(checkboxes_frame_, text='Show SSIM Rate Distortion', variable=show_ssim_rate_var, command=lambda: [show_psnr_rate_var.set(False),show_brisque_rate_var.set(False), update_quality_metrics_rate_plot()]).pack(side=tk.LEFT)
    ttk.Checkbutton(checkboxes_frame_, text='Show BRISQUE Rate Distortion', variable=show_brisque_rate_var, command=lambda: [show_psnr_rate_var.set(False),show_ssim_rate_var.set(False), update_quality_metrics_rate_plot()]).pack(side=tk.LEFT)

    update_quality_metrics_rate_plot()


    ################################ energy ###########################################################

    add_capture_energy_var = tk.BooleanVar(value=False)    

    def update_energy_plots():
        if  add_capture_energy_var.get():
            ploting.plot_captured_energy(ax_enrgy,st_frame_data)
        else :
            ploting.plot_energy(ax_enrgy,st_frame_data)

    add_capture_energy_frame = ttk.Frame(energy_tab)
    add_capture_energy_frame.pack(side=tk.TOP, fill=tk.X)
    ttk.Checkbutton(add_capture_energy_frame, text='Add Capture Energy', variable=add_capture_energy_var, command=update_energy_plots).pack(side=tk.LEFT)
    update_energy_plots()




    show_signal_lost_var = tk.BooleanVar(value=True)
    show_ber_var = tk.BooleanVar(value=False)
    show_snr_var = tk.BooleanVar(value=False)

    def update_noise_plots():
        if show_signal_lost_var.get():
            ploting.plot_signal_lost(ax_noise, st_packet_data)
        elif show_ber_var.get():
            ploting.plot_ber(ax_noise, st_packet_data)
        elif show_snr_var.get():
            ploting.plot_snr(ax_noise,st_packet_data)

    checkboxes_noise_frame = ttk.Frame(noise_tab)
    checkboxes_noise_frame.pack(side=tk.TOP, fill=tk.X)

    ttk.Checkbutton(checkboxes_noise_frame, text='Show Signal Lost', variable=show_signal_lost_var, command=lambda: [show_ber_var.set(False), show_snr_var.set(False), update_noise_plots()]).pack(side=tk.LEFT)
    ttk.Checkbutton(checkboxes_noise_frame, text='Show BER', variable=show_ber_var, command=lambda: [show_signal_lost_var.set(False),show_snr_var.set(False), update_noise_plots()]).pack(side=tk.LEFT)
    ttk.Checkbutton(checkboxes_noise_frame, text='Show SNR', variable=show_snr_var, command=lambda: [show_signal_lost_var.set(False), show_ber_var.set(False), update_noise_plots()]).pack(side=tk.LEFT)
    update_noise_plots()


    notebook.pack(expand=True, fill='both')

    results_view.mainloop()

# Function to load images
def load_images(tab, directory):
    if not os.path.exists(directory):
        messagebox.showerror("Error", f"Directory '{directory}' does not exist.")
        return

    canvas = tk.Canvas(tab)
    v_scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    images = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def arrange_images():
        for widget in scrollable_frame.winfo_children():
            widget.destroy()

        max_cols = max(1, tab.winfo_width() // 144)

        row = 0
        col = 0

        for image_file in images:
            image_path = os.path.join(directory, image_file)
            img = Image.open(image_path)
            img.thumbnail((144, 144))
            img = ImageTk.PhotoImage(img)

            img_label = tk.Label(scrollable_frame, image=img)
            img_label.image = img
            img_label.grid(row=row * 2, column=col, padx=5, pady=5)

            name_label = tk.Label(scrollable_frame, text=image_file)
            name_label.grid(row=row * 2 + 1, column=col, padx=5, pady=5)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    arrange_images()
    tab.bind("<Configure>", lambda event: arrange_images())

    canvas.grid(row=0, column=0, sticky="nsew")
    v_scrollbar.grid(row=0, column=1, sticky="ns")
    h_scrollbar.grid(row=1, column=0, sticky="ew")

    tab.grid_rowconfigure(0, weight=1)
    tab.grid_rowconfigure(1, weight=0)
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_columnconfigure(1, weight=0)

# Function to create a tab
def create_tab(notebook, tab_name, file_path):
    frame = ttk.Frame(notebook)
    text_widget = ScrolledText(frame, wrap=tk.WORD)
    text_widget.pack(expand=True, fill='both')
    lines = load_file(file_path)
    colorize_text(text_widget, lines)
    text_widget.config(state=tk.DISABLED)
    notebook.add(frame, text=tab_name)

def open_results_view(root):
    root.destroy()
    init_results_view()

##############################################################################
################################Ploting#######################################

def get_data():
    st_packets_data = pd.read_csv(os.path.join(Parameters.trace_file_path, 'st-packet'), sep = '\t', comment='#', header=None)
    st_packets_data.columns =  ["time", "seqNb", "pktSize", "frameNb", "frameType", "layerNb", "blocksList","signal_lost(db)","SNR","BER"] 
    
    st_frame_data = pd.read_csv(os.path.join(Parameters.trace_file_path, 'st-frame'), sep = '\t', comment='#', header=None)
    st_frame_data.columns = ["Rank", "Type", "Size(Bytes)", "refPSNR", "refSSIM", 'refBRISQUE', "bpp", "layers Size (bits)",
                           "captureEnergy(mJ)", "encodingEnergy(mJ)", "bit rate (kbps)"]
    rt_frame_data = pd.read_csv(os.path.join(Parameters.trace_file_path, 'rt-frame'), sep = '\t', comment='#', header=None)
    rt_frame_data.columns = ["Rank", "frameType", "PSNR", "SSIM","BRISQUE"]

    return {
        "st_packet":st_packets_data,
        "st_frame":st_frame_data,
        "rt_frame":rt_frame_data
    }





def open_root_view(root):
    root.mainloop()





if __name__ == "__main__":
    root = init_root_view()
    open_root_view(root)