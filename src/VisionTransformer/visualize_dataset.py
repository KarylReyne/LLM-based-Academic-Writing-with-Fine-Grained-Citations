import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
import sys  # For command-line arguments

class HDF5ImagePlayer:
    def __init__(self, hdf5_file):
        # Open the HDF5 file
        self.f = h5py.File(hdf5_file, 'r')
        self.rgb_images = self.f['/rgb_images']
        self.num_images = self.rgb_images.shape[0]

        # Set up the Matplotlib figure and axis
        self.fig, self.ax = plt.subplots()
        self.im = None

        # Connect key press and close events
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)

        # Start playing all images
        self.play_images()
        plt.show()

    def get_images(self):
        # Retrieve and decode all images
        images = []
        for idx in range(self.num_images):
            img_bytes = self.rgb_images[idx]
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
            print(img_array.shape)
            images.append(img_array)
        return images

    def play_images(self):
        # Play all images as an animation
        self.images = self.get_images()
        if self.im is None:
            self.im = self.ax.imshow(self.images[0])
            self.ax.axis('off')
        else:
            self.im.set_data(self.images[0])
        self.anim = FuncAnimation(
            self.fig,
            self.update_frame,
            frames=len(self.images),
            interval=50,  # Adjust the interval as needed
            repeat=True
        )
        self.anim.running = True
        self.fig.canvas.draw_idle()

    def update_frame(self, frame):
        # Update the image for each frame
        self.im.set_data(self.images[frame])
        print(f"playing frame: {frame}")
        return [self.im]

    def on_press(self, event):
        # Handle key press events
        if event.key == ' ':
            # Pause or resume the animation
            if self.anim.running:
                self.anim.event_source.stop()
                self.anim.running = False
            else:
                self.anim.event_source.start()
                self.anim.running = True
        elif event.key == 'escape':
            # Exit the GUI
            plt.close(self.fig)

    def on_close(self, event):
        # Close the HDF5 file when the GUI is closed
        self.f.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <hdf5_file>")
        sys.exit(1)
    hdf5_file = sys.argv[1]
    player = HDF5ImagePlayer(hdf5_file)

