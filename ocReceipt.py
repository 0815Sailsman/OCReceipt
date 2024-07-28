import cv2 as cv
import numpy as np
import math
from wobbly_lines import algo

def main():
    # Load img
    img = cv.imread("./receipt.jpg")

    # Contrast preprocessing
    for row in range(0,img.shape[1]):
        for col in range(0,img.shape[0]):
            perceived_brightness = calc_perceived_brightness_of_bgr_pixel(img[col, row])
            if (perceived_brightness < 100):
                img[col,row]=np.array([0, 0, 0])
            else:
                img[col,row]=np.array([255, 255, 255])

    #cv.imwrite('./contrasted.jpg', rotate(img2, 90))

    rows_to_color = []
    # check every row by itself as a starting point
    early_exits = 0
    for row_index in range(0, img.shape[1]):
        result = algo(row_index, img)
        if result == -1:
            print("early exit :)")
            early_exits += 1
            continue
        if result is None:
            continue
        rows_to_color.append(result)

    print(str(early_exits) + " early exits")
    color_img_by_instructions(img, rows_to_color)

    # display(img2)
    cv.imwrite('./line-contrast.png', rotate(img, 90))
    
    # extract rows / items
    # Preprocess rows / items for OCR preprocessen
    # OCR

def color_img_by_instructions(img2, rows_to_color):
    for obj in rows_to_color:
        start = obj[0]
        instructions = obj[1]
        print(obj[0])
        print(len(instructions))
        offset = 0
        counter = 0
        # For every 5 pixel wide column
        # for col_block in range(0, (img2.shape[0] // 5 - 1)):
        for col_block in range(0, img2.shape[0] - 1):
            # For every pixel in that block
#            for i in range(5):
            for i in range(1):
                # Break if out of bound
                if start + offset >= img2.shape[1]:
                    continue

                #img2[col_block * 5 + i, start + offset] = np.array([0, 0, 255])
                img2[col_block, start + offset] = np.array([0, 0, 255])
            if counter < len(instructions):
                offset += instructions[counter]
                counter += 1



def calc_perceived_brightness_of_bgr_pixel(pixel):
    return pixel[2] * 0.2126 + pixel[1] * 0.7152 + pixel[0] * 0.0722

def contrast(img):
    lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((img, enhanced_img))
    display(enhanced_img)

def display(img):
    cv.imshow("Display window", display_scale(img))
    k = cv.waitKey(0) # Wait for a keystroke in the window

def display_scale(img):
    return cv.resize(rotate(img, 90), (850, 1000), interpolation= cv.INTER_LINEAR)

def rotate(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

def stuff():
    print(iter1)
    iter2 = []
    for inx in range(len(iter1)):
        if inx < (len(iter1) - 2):
            if (iter1[inx+1] - iter1[inx]) > 10:
                iter2.append(iter1[inx])
    print(iter2)
    
    iter3 = []
    for hinx in range(len(iter2)-2):
        iter3.append(iter2[hinx+1] - iter2[hinx])
    height = int(sum(iter3) / len(iter3))
    print(height)
    temp = 0
    while temp < img2.shape[1]:
        for row in range(int(height)):
            for col in range(0,img2.shape[0]):
                if temp + row < img2.shape[1]:
                    a = img2[col, temp + row]
                    inc = a[0]+ 100
                    img2[col, temp + row] = np.array([inc if inc <= 255 else 255, a[1], a[2]])
                else:
                    cv.imwrite('./line-contrast.png', img2)
                    exit(0)
        temp += (height + 5)



if __name__ == "__main__":
    main()
