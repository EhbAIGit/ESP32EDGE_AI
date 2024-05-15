#!/usr/bin/python

# author: Lode 2021

from pathlib import Path
import statistics
import requests
from PIL import ImageTk, Image, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import time
import datetime 
import os 
import sys
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


CURRENT_DIRECTORY = Path.cwd()

########### DEFAULT VALUES ###########
DEFAULT_PICTURES_PATH = Path(CURRENT_DIRECTORY, "pics")
DEFAULT_OUTPUT_PATH = Path(CURRENT_DIRECTORY, "output")
COLOR_PICKER_SIZE = 3
IMAGE_TAKEOUT_SIZE = 300
DEFAULT_DATASET_FILE = "reference_data.csv"
trained_model = None
#######################################

clicked_coords = None
reference_R = 0
reference_G = 0
reference_B = 0

def get_user_path(default_folder, use_default_path=False): 
    if use_default_path:
        return default_folder
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_path = filedialog.askdirectory(title="Select Folder", initialdir=default_folder)
    root.destroy()
    
    return user_path or default_folder

def obtain_mouse_coordinate_independent(picture_path):    
    root = tk.Tk()

    def printcoords(event):
        global clicked_coords 
        clicked_coords = (event.x, event.y)
        root.destroy()
    
    canvas = tk.Canvas(root, width=800, height=600)
    canvas.pack(expand=tk.YES, fill=tk.BOTH)
    
    # Load the image
    img = Image.open(picture_path)
    img.thumbnail((800, 600))  # Resize image if needed
    photo_img = ImageTk.PhotoImage(img)
    
    # Display the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
    canvas.image = photo_img  # Keep a reference to prevent garbage collection

    canvas.bind("<ButtonPress-1>", printcoords)
    root.mainloop()


def capture_image(urlCapture, urlGetPicture, picture_path):
    
    r = requests.get(urlCapture, allow_redirects=True)
    time.sleep(5)
    r = requests.get(urlGetPicture, allow_redirects=True)
    fileLocation = picture_path
    open(fileLocation, 'wb').write(r.content)
    return fileLocation

def save_reference_data(image_path, clicked_coords, dataset_file_path):
    csv_line = f"{image_path},{clicked_coords[0]},{clicked_coords[1]}"
    add_to_results_csv2(dataset_file_path, csv_line)
    


def process_reference_picture(picture_path, dataset_file_path, output_folder, image_tag, picture_folder):
    global reference_R, reference_G, reference_B
    obtain_mouse_coordinate_independent(picture_path)
        # Load the image
    image = Image.open(picture_path)

    # Get the region of interest
    box_size = 10
    scaled_coords = (
        clicked_coords[0] * (image.width / 800),
        clicked_coords[1] * (image.height / 600)
    )

    box = (
        scaled_coords[0] - box_size // 2,
        scaled_coords[1] - box_size // 2,
        scaled_coords[0] + box_size // 2,
        scaled_coords[1] + box_size // 2
    )
    region_of_interest = image.crop(box)

    region_filename = f"{image_tag}_region_of_interest.jpg"
    color_swatch_filename = f"{image_tag}_median_color_swatch.jpg"

    # Save the region of interest

    output_path_region = Path(output_folder) / region_filename
    region_of_interest.save(output_path_region)

    # Calculate median RGB values
    R_list, G_list, B_list = zip(*list(region_of_interest.getdata()))
    median_R = int(statistics.median(R_list))
    median_G = int(statistics.median(G_list))
    median_B = int(statistics.median(B_list))

    # Update reference values if it's the reference image
    if picture_path == Path(picture_folder) / "reference.jpg":
        reference_R = median_R
        reference_G = median_G
        reference_B = median_B

    # Create a median color swatch image
    median_color_swatch = Image.new('RGB', (100, 100), (median_R, median_G, median_B))
    output_path_color_swatch = Path(output_folder) / color_swatch_filename
    median_color_swatch.save(output_path_color_swatch)

    save_reference_data(picture_path, clicked_coords, dataset_file_path)

    return {
        "median_R": median_R,
        "median_G": median_G,
        "median_B": median_B
    }
    


def train_model(dataset_file):
    with open(dataset_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        coordinates = []
        for row in reader:
            x_coord, y_coord = int(row[1]), int(row[2])
            coordinates.append((x_coord, y_coord))
    
    # Calculate centroid
    if coordinates:
        x_coords, y_coords = zip(*coordinates)
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        return centroid_x, centroid_y
    else:
        return None  # Return None if dataset is empty




def process_picture(picture_path, output_folder, image_tag, picture_folder):
    global reference_R, reference_G, reference_B, trained_model

    # Load the image
    image = Image.open(picture_path)

    # Get the region of interest
    box_size = 10
    scaled_coords = (
        trained_model[0] * (image.width / 800),
        trained_model[1] * (image.height / 600)
    )

    box = (
        scaled_coords[0] - box_size // 2,
        scaled_coords[1] - box_size // 2,
        scaled_coords[0] + box_size // 2,
        scaled_coords[1] + box_size // 2
    )
    region_of_interest = image.crop(box)

    region_filename = f"{image_tag}_region_of_interest.jpg"
    color_swatch_filename = f"{image_tag}_median_color_swatch.jpg"

    # Save the region of interest

    output_path_region = Path(output_folder) / region_filename
    region_of_interest.save(output_path_region)

    # Calculate median RGB values
    R_list, G_list, B_list = zip(*list(region_of_interest.getdata()))
    median_R = int(statistics.median(R_list))
    median_G = int(statistics.median(G_list))
    median_B = int(statistics.median(B_list))

    # Update reference values if it's the reference image
    if picture_path == Path(picture_folder) / "reference.jpg":
        reference_R = median_R
        reference_G = median_G
        reference_B = median_B

    # Create a median color swatch image
    median_color_swatch = Image.new('RGB', (100, 100), (median_R, median_G, median_B))
    output_path_color_swatch = Path(output_folder) / color_swatch_filename
    median_color_swatch.save(output_path_color_swatch)

    return {
        "median_R": median_R,
        "median_G": median_G,
        "median_B": median_B
    }


def add_to_results_csv(filepath, csv_line):
    if not Path(filepath).is_file():
        with open(filepath, "w") as f:
            f.write("ID, Unic_number, Date, Red, Green, Blue, Ref. Red,Ref. Green,Ref. Blue\n")
    
    with open(filepath, "a") as f:
        f.write(csv_line + "\n")

def add_to_results_csv2(filepath, csv_line):
    if not Path(filepath).is_file():
        with open(filepath, "w") as f:
            f.write("Image_Path, Clicked_X, Clicked_Y\n")
    
    with open(filepath, "a") as f:
        f.write(csv_line + "\n")

def check_network_connection(host):
    response = os.system(f"ping -n 1 {host}")  # Windows: Use -n
    return response == 0


def main():
    if not check_network_connection('192.168.4.1'):
        messagebox.showerror("Network Error", "Please connect to the correct Wi-Fi network: \n ID: PELLET \n Password: password")
        sys.exit(1) # Return None to indicate failure
    global trained_model


    user_confirmation = messagebox.askokcancel("Pictures Folder", "Next window will ask you where to save your pictures. \n Convention: Make a new folder in the outputmap with the name of the experiment")
    if user_confirmation is False:  # Check if user clicked Cancel
        return  # Stop execution if Cancel was clicked
        
    pictures_folder = get_user_path(DEFAULT_PICTURES_PATH, False)

    user_confirmation = messagebox.askokcancel("Output Folder", "Next window will ask you where to save your main results. \n This is where the csv file is with others results, if you have already. \n Convention: Select the folder where the different folders of the pictures are. ")
    if user_confirmation is False:  # Check if user clicked Cancel
        return  # Stop execution if Cancel was clicked
    output_folder = get_user_path(DEFAULT_OUTPUT_PATH, False)


    dataset_file_path = os.path.join(output_folder, DEFAULT_DATASET_FILE)



    csv_path = Path(output_folder, "results.csv")
    csv_path2 = Path(pictures_folder, "results.csv")

    if csv_path2.is_file():
        user_input = messagebox.askquestion("File Exists", "A results.csv file already exists. Delete it?")
        if user_input == "yes":
            csv_path.unlink()

    urlCapture = 'http://192.168.4.1/capture'
    urlGetPicture = 'http://192.168.4.1/saved-photo'

    user_confirmation = messagebox.askokcancel("Reference Pellet", "Next window is a picture of your reference pellet. \n The picture will be taken when your press OK. \n Are you sure to proceed?")
    if user_confirmation is False:  # Check if user clicked Cancel
        return  # Stop execution if Cancel was clicked

    picture_path = Path(pictures_folder) / f"reference.jpg"
    capture_image(urlCapture, urlGetPicture, picture_path)
    process_reference_picture(picture_path,  dataset_file_path, output_folder, picture_path, pictures_folder )

    messagebox.showinfo("Training Model", "Training the model with collected reference data...")
    trained_model = train_model(dataset_file_path)
    messagebox.showinfo("Model Trained", "Model training completed.")

    while True:
        user_input = simpledialog.askstring(
            "Image Tag",
            "Please provide an image tag. \n "
            "Naming convention is like this: pelletname_reagent_concentration \n "
            "Place Your Pellet and press OK if your pellet is in place. \n "
            "Press Cancel if you want to end this session. \n"
        )
        if user_input is None:  # Check if user clicked Cancel
            break  # Exit the loop if Cancel was clicked
        

        picture_path = Path(pictures_folder) / f"{user_input}.jpg"
        capture_image(urlCapture, urlGetPicture, picture_path)
        processed_data = process_picture(picture_path, output_folder, picture_path, pictures_folder)

        current_datetime = datetime.datetime.now()

# Format the datetime as desired (e.g., YYYYMMDD_HHMM)
        formatted_datetime = current_datetime.strftime("%d-%m-%Y %H:%M")

        # Save processed data to CSV
        csv_line = f"{user_input},{picture_path},{formatted_datetime},{processed_data['median_R']},{processed_data['median_G']},{processed_data['median_B']},{reference_R},{reference_G},{reference_B}"
        add_to_results_csv(csv_path, csv_line)
        add_to_results_csv(csv_path2, csv_line)

    messagebox.showinfo("Finished", "Processing completed. \n Session is ended. \n  Don't forget to dubble check your results. ")

if __name__ == "__main__":
    main()