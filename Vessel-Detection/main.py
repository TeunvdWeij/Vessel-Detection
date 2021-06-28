import pytesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import cv2
import pandas as pd

def get_characters(img_src="Images/alaska-warrior.jpg", show_img=False, show_thresh=False):

    if show_img:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 600)

    if show_thresh:
        cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('thresh', 800, 600)

    image = cv2.imread(img_src)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # OCR
    results = pytesseract.image_to_string(255 - thresh, lang='eng', config='--psm 6')

    if show_img:
        cv2.imshow('image', image)

    if show_thresh:
        cv2.imshow('thresh', thresh)

    if show_img or show_thresh:
        cv2.waitKey()
        cv2.destroyAllWindows()

    return results

def check_if_listed(result_string, vessel_names):
    for name in vessel_names:
        if name.lower() in result_string.lower():
            print('Vessel is on the IUUList')
            return

    print("Vessel is not on the IUUList")
    return 

df = pd.read_excel('IUU_data/IUUList-20210628.xls',
                 sheet_name='IUUList', usecols='B')
vessel_names = df['Name']

results = get_characters()
check_if_listed(results, vessel_names)

