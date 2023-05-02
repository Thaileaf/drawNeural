import tkinter as tk
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from joblib import load


# Calculate the resize factor
resize_factor = 28 / 256
#degree = 2
#poly = PolynomialFeatures(degree)

model_log = load('./BestLogistic.joblib') 

model_svm = load('./BestSVM2.joblib') 
model_neural = load('./BestNeural2.joblib')

model = model_neural
class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("Drawing App")

        # Create a frame widget with a 1000x1000 size
        self.frame = tk.Frame(master, bg="#FFD139", width=700, height=700)
        self.frame.pack()   

        
        self.label = tk.Label(root, text="Quick Draw!", bg="#FFD139", font=("Arial", 24))

        self.label.place(relx=0.5, rely=0.2, anchor="center")
        self.my_lambda = lambda model: self.setModel(model)


        self.current_model = model_log
        self.button1 = tk.Button(master, text="Logistic", command=self.my_lambda(model_log))
        self.button1.pack(side=tk.LEFT)

        self.button2 = tk.Button(master, text="SVM", command=lambda: self.my_lambda(model_svm))
        self.button2.pack(side=tk.LEFT)

        self.button3 = tk.Button(master, text="Neural", command=lambda: self.my_lambda(model_neural))
        self.button3.pack(side=tk.LEFT) 

        self.current_model_name = "Logistic"

        # Create a canvas widget with a 256x256 size
        self.canvas = tk.Canvas(master, width=256, height=256, bg="white")
        self.canvas.place(relx=0.5, rely=0.5, anchor="center")

        # Create a Text widget for displaying output underneath the canvas
        self.output = tk.Text(self.frame, width=50, height=5, bg="white")
        self.output.place(relx=0.5, rely=0.8, anchor="center")

        # Bind the left mouse button to draw on the canvas
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Initialize the list of pixel colors to None
        self.pixels = [[0 for j in range(256)] for i in range(256)]

        self.start_clock()

    def setModel(self, model):
        self.current_model = model
        if model == model_log:
            self.current_model_name = "Logistic"
        elif model == model_svm:
            self.current_model_name = "SVM"
        elif model == model_neural:
            self.current_model_name = "Neural"
    def start_clock(self):
         # Convert the list of pixels into a NumPy array
        pixels_np = np.array(self.pixels)

    

        # Resize the NumPy array to 28x28 using the zoom function
        resized_np = zoom(pixels_np, resize_factor, order=1).reshape(1, 784)  # order=1 for bilinear interpolation
        # print(resized_np)
        # print(resized_np.shape)
        #resized_poly = poly.fit_transform(resized_np)
        model = self.current_model
        y_pred = model.predict(resized_np)
        animal = "?"
        print(y_pred)
        if(y_pred[0] == 0):
            animal = "giraffe"
        elif(y_pred[0] == 1):
            animal = "sheep"
        elif(y_pred[0] == 2):
            animal = "cat"

        
        self.output.insert("1.0", f"{self.current_model_name}:{animal}\n")

        self.master.after(1000, self.start_clock)

    def in_bounds(self, event):
        return 0 <= event.x <= 256 and 0 <= event.y <= 256

    def start_draw(self, event):
        # Create a new line at the mouse position
        if self.in_bounds(event):
            self.line = self.canvas.create_line(event.x, event.y, event.x, event.y, width=3, fill="black")

    def draw(self, event):
        if self.in_bounds(event):
            # Extend the line to the new mouse position
            self.canvas.coords(self.line, *self.canvas.coords(self.line), event.x, event.y) # Get the color of the pixel at the mouse position
            self.pixels[event.x][event.y] = 256
            # print(event.x, event.y)
            





# Create a tkinter window and start the main loop
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()


