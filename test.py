from tkinter.filedialog import askopenfilename
import open 
file=askopenfilename(filetypes=[("jpg files","*.jpg"),("png files","*.png")])
print("file name",file)