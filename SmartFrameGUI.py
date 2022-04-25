import Tkinter as tk
import numpy as np
from PIL import Image, ImageTk


class SmartFrameGUI:

    def __init__(self, height=480, width=800):

        # the shape of the display
        self._shape = (height, width)

        # create the structure
        self._root = tk.Tk()
        self._root.attributes('-zoomed', True)  # This just maximizes it so we can see the window. It's nothing to do with fullscreen.

        self._frame = tk.Frame(self._root)
        self._frame.config(cursor='none')
        self._frame.pack(side='top', fill='both', expand='yes')

        self._canvas = tk.Canvas(self._frame)  # , background='#000fff000')
        self._canvas.pack(side='top', fill='both', expand='yes')

        # create the space for the image
        self._canvas.image = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)))
        self._canvas.image_ref = self._canvas.create_image(0, 0, image=self._canvas.image, anchor='nw')

        # binding of functions
        self._root.bind("<F11>", self.toggle_fullscreen)
        self._root.bind("<Escape>", self.exit)
        self._root.bind("<Button-1>", self.toggle_fullscreen)

        # save the state of the gui, if it is in full screen
        self._fullscreen_state = False

        # start in full screen
        self.toggle_fullscreen()

    def display_image(self, img, rotation=0):
        if rotation == 0:
            self._canvas.image = ImageTk.PhotoImage(image=Image.fromarray(img))
        elif rotation == 1:
            self._canvas.image = ImageTk.PhotoImage(image=Image.fromarray(np.rot90(img, axes=(0, 1))))
        elif rotation == -1:
            self._canvas.image = ImageTk.PhotoImage(image=Image.fromarray(np.rot90(img, axes=(1, 0))))
        elif rotation == 2:
            self._canvas.image = ImageTk.PhotoImage(image=Image.fromarray(np.rot90(np.rot90(img, axes=(1, 0)), axes=(1, 0))))
        else:
            self._canvas.image = ImageTk.PhotoImage(image=Image.fromarray(img))

        self._canvas.itemconfig(self._canvas.image_ref, image=self._canvas.image)

    def refresh(self):
        # self._root.update_idletasks()
        self._root.update()

    def toggle_fullscreen(self, event=None):
        self._fullscreen_state = not self._fullscreen_state  # Just toggling the boolean
        self._root.attributes("-fullscreen", self._fullscreen_state)

    def exit(self, event=None):
        self._root.destroy()


if __name__ == '__main__':
    import os
    import time
    import dlib

    # Directory where to find the photos to use
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')

    # example program that reads the photos in the example folder and calculate the average of all the faces
    images = [dlib.load_rgb_image(os.path.join(examples_dir, image)) for image in os.listdir(examples_dir)]

    gui = SmartFrameGUI()
    for img in images + images:
        gui.display_image(img, 1)
        gui.refresh()
        time.sleep(2)
