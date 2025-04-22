'''
This script includes a function which converts pdf in a given 
folder path into images. 
For every pdf, a dictionary entry for a json is being generated 
and all image-names of the PDF are saved within the key 'pages', which 
value's a list. The images themselves are for now saved in the same folder
as the PDFs.
'''

import fitz  # PyMuPDF
from PIL import Image
import os
import json

def convert_pdf_to_images(filename, pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    images = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=dpi)
        img_path = f"./testdaten/{filename}_page_{i+1}.png"
        pix.save(img_path)
        images.append(img_path)
    return images


def run():
    path_folder_with_data = 'C:/Users/imgey/Desktop/Arbeit/DFKI Berlin/Projekte/DRV/data/testdaten'

    documents = list()

    for filename in os.listdir(path_folder_with_data):
        if filename.endswith('.pdf'):

            image_names = convert_pdf_to_images(filename, f"./testdaten/{filename}")

            # create a dict for the file
            document = {
                'filename': filename,
                'pages': image_names
            }
            documents.append(document)
    
    with open('./testdaten/data.json', 'w', encoding='UTF-8') as f:
        json.dump(documents, f, indent=4)


run()
