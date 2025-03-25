import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Tkinter Test")

# Create a label
label = tk.Label(root, text="Hello, Tkinter is working!", font=("Arial", 14))
label.pack(pady=20)

# Create a button to close the window
button = tk.Button(root, text="Close", command=root.destroy)
button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
