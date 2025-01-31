import argparse
import math
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_w', type=int, default=512, help='Cell width')
    parser.add_argument('--cell_h', type=int, default=512, help='Cell height')
    parser.add_argument('--cols', type=int, default=3, help='Number of columns')
    parser.add_argument('--imgs', nargs='+', type=str, help='Paths to img files')
    parser.add_argument('--names', nargs='+', type=str, help='Labels for img files')

    args = parser.parse_args()

    return args

class ImageComparisonApp:
    def __init__(self, root, image_dict, cell_width, cell_height, cols):
        self.root = root
        self.image_dict = image_dict
        self.keys = list(image_dict.keys())

        values = list(image_dict.values())

        self.max_index = len(values[0])
        self.relativeScaleFactors = []

        self.current_index = 0
        self.zoom_factor = 1.0

        rows = (self.max_index-1) // cols

        self.images_frame = tk.Frame(root, width=cell_width * cols, height=cell_height * rows)
        self.images_frame.pack_propagate(False)
        self.images_frame.pack()

        self.cell_width = cell_width 
        self.cell_height = cell_height

        self.cols = cols

        self.controls_frame = tk.Frame(root, width=cell_width * cols, height=150)
        self.controls_frame.pack()

        self.left_shift = 0
        self.down_shift = 0

        self.repeat = False

        self.labels = []
        self.tk_images = []

        for i in range(len(self.keys)):
            label = tk.Label(self.images_frame, text=self.keys[i], compound=tk.TOP)
            self.labels.append(label)

        self.zoom_in_button = tk.Button(self.controls_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT)

        self.zoom_out_button = tk.Button(self.controls_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.controls_frame, text="Next", command=self.next_images)
        self.next_button.pack(side=tk.LEFT)

        self.prev_button = tk.Button(self.controls_frame, text="Previous", command=self.prev_images)
        self.prev_button.pack(side=tk.LEFT)

        self.shift_left_button = self.create_repeatable_button(self.controls_frame, text="<", command=self.shift_left)
        self.shift_left_button.pack(side=tk.LEFT)

        self.shift_up_button = self.create_repeatable_button(self.controls_frame, text="^", command=self.shift_up)
        self.shift_up_button.pack(side=tk.LEFT)

        self.shift_down_button = self.create_repeatable_button(self.controls_frame, text="v", command=self.shift_down)
        self.shift_down_button.pack(side=tk.LEFT)

        self.shift_right_button = self.create_repeatable_button(self.controls_frame, text=">", command=self.shift_right)
        self.shift_right_button.pack(side=tk.LEFT)

        self.update_images()

    def create_repeatable_button(self, parent, text, command):
        def start_repeat():
            command()  # Perform the action immediately
            self.repeat_flag = True
            self.repeat_action(command)

        def stop_repeat(event=None):
            self.repeat_flag = False

        button = tk.Button(parent, text=text)
        button.bind("<ButtonPress>", lambda event: start_repeat())
        button.bind("<ButtonRelease>", stop_repeat)
        return button

    def repeat_action(self, command):
        if self.repeat_flag:
            command()
            self.root.after(50, lambda: self.repeat_action(command))

    def update_images(self):
        # Clear the previous images
        for i, label in enumerate(self.labels):
            label.image = None

        # Get the current set of images
        images = []
        for key in self.keys:
            images.append(self.image_dict[key][self.current_index])

        cols = self.cols

        self.tk_images = []
        for img in images:
            cell_width = self.cell_width
            cell_height = self.cell_height
            aspect_ratio = img.width / img.height

            if cell_width / cell_height > aspect_ratio:
                new_height = cell_height
                new_width = int(cell_height * aspect_ratio)
            else:
                new_width = cell_width
                new_height = int(cell_width / aspect_ratio)

            resized_img = img.resize((int(new_width * self.zoom_factor), int(new_height * self.zoom_factor)))
            cropped_img = resized_img.crop((self.left_shift, self.down_shift, cell_width+self.left_shift, cell_height+self.down_shift))

            tk_img = ImageTk.PhotoImage(cropped_img)
            self.tk_images.append(tk_img)

        for i, tk_img in enumerate(self.tk_images):
            self.labels[i].config(image=tk_img)
            self.labels[i].grid(row=i // cols, column=i % cols, padx=5, pady=5)
            self.labels[i].image = tk_img

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_images()
    
    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.update_images()

    def next_images(self):
        self.current_index = (self.current_index + 1) % self.max_index
        self.update_images()

    def prev_images(self):
        self.current_index = (self.current_index - 1) % self.max_index
        self.update_images()

    def shift_left(self):
        self.left_shift = self.left_shift - 10
        self.update_images()

    def shift_right(self):
        self.left_shift = self.left_shift + 10
        self.update_images()
    
    def shift_up(self):
        self.down_shift = self.down_shift - 10
        self.update_images()
    
    def shift_down(self):
        self.down_shift = self.down_shift + 10
        self.update_images()

if __name__ == "__main__":
    args = get_args()

    print("Images taken from files in: " + ",".join(args.imgs))
    print("Labelled as: " + ",".join(args.names))
    

    image_dict = {}
    if len(args.imgs) != len(args.names):
        print("Unequal length of img arrays and names! Trimming arrays")

    for i in range(min(len(args.imgs),len(args.names))):
        images = sorted(os.listdir(args.imgs[i]))
        images_PIL = [
            Image.open(os.path.join(args.imgs[i], image)) 
            for image in images
        ]

        print(f'Added to dictionary under {args.names[i]}: {",".join(images)}')

        image_dict[args.names[i]] = images_PIL

    root = tk.Tk()
    app = ImageComparisonApp(root, image_dict, args.cell_w, args.cell_h, args.cols)
    root.mainloop()