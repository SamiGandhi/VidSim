# start.py
# Entry-point to launch VidSim core pipeline
from tkinter import messagebox
from ui import UI as ui
from _version import __version__

def start() -> None:
     print(f"Starting VidSim (Version: {__version__})")
     root = ui.init_root_view()
     ui.open_root_view(root)

if __name__ == "__main__":
     try: 
          start()
     except Exception as e:
          messagebox.showerror("Error", str(e))
          start()

     
