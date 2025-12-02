import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText






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
            text_widget.tag_config(str(i), foreground="bleu")
        elif 'R' in line:
            text_widget.tag_add(str(i), f"{i+1}.0", f"{i+1}.end")
            text_widget.tag_config(str(i), foreground="purple")

def create_tab(notebook, tab_name, file_path):
    frame = ttk.Frame(notebook)
    text_widget = ScrolledText(frame, wrap=tk.WORD)
    text_widget.pack(expand=True, fill='both')
    lines = load_file(file_path)
    colorize_text(text_widget, lines)
    text_widget.config(state=tk.DISABLED)
    notebook.add(frame, text=tab_name)

def  init_results_view(st_packet,st_frame,rt_frame):

    results_view = tk.Tk()
    root.title("Results")
    icon = tk.PhotoImage(file=r'src\res\icon.png')
    results_view.iconphoto(True, icon)
    results_view.title("Set Parameters")
    tool_tip = Pmw.Balloon(results_view)
    notebook = ttk.Notebook(results_view)

    trace_tab = ttk.Notebook(notebook)
    trace_frame = ttk.Frame(trace_tab)

    create_tab(trace_tab, "st_packets", st_packet)
    create_tab(trace_tab, "st_frames", st_frame)
    create_tab(trace_tab, "rt_frames", rt_frame)

    trace_tab.pack(expand=True, fill='both')
    notebook.add(trace_tab, text="Trace")

    plots_frame = ttk.Frame(notebook)
    notebook.add(plots_frame, text="Plots")

    frames_frame = ttk.Frame(notebook)
    notebook.add(frames_frame, text="Frames")

    notebook.pack(expand=True, fill='both')

    return results_view

