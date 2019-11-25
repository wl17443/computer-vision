import numpy as np
import cv2

sample = ["dart4.jpg", "dart5.jpg", "dart13.jpg", "dart14.jpg", "dart15.jpg"]
ground_truth = [[(351, 99, 117,173)],[(65,148,56,58), (253,171,48,64), (379,195,62,53), (513,179,60,64), (652,190,49,59), (54,251,60,70), (194,215,54,68), (296,238,50,71), (427,234,56,72), (562,250,59,65), (682,244,48,68)], [(425,120,101,130)], [(473,214,75,104), (727,189,98,103)],[] ]

cascade = cv2.CascadeClassifier('frontalface.xml')

def detectAndDisplay( img_file, counter ):
    # load input image in grayscale mode
    img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale( gray_img, 1.1, 1)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    for (x,y,w,h) in ground_truth[counter]:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    print(counter + 1 , " - Faces found: ", len(faces), "Expected: ", len(ground_truth[counter]))

    tpos = 0
    for face in faces:
        for actualFace in ground_truth[counter]:
            result = intersectionOverUnion( face, actualFace )
            if result > 0:
                tpos += 1

    positives = len(faces)
    falseNegative = len(ground_truth[counter]) - tpos

    f1 = f1Score(tpos, positives, falseNegative)
    tpr = tpos / positives

    print("True Positive Rate: ", tpr, "True Positives: ", tpos, "Positives: ", len(faces), "F1-Score: ", f1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def intersectionOverUnion( box1, box2 ):
    (x1, y1, w1, h1) = box1
    (x2, y2, w2, h2) = box2

    left = np.maximum(x1, x2)
    right = np.minimum(x1+w1, x2+w2)
    top = np.maximum(y1, y2)
    bottom = np.minimum(y1+h1, y2+h2)

    if left > right:
        intersec = 0
    elif bottom < top:
        intersec = 0
    else:
        intersec = (right-left) * (bottom-top)

    # Calculate the area of union
    union = w1 * h1 + w2 * h2 - intersec

    # Calculate the IOU/Jaccard Index
    jaccard = intersec / union

    if jaccard < 0.6:
        return 0
    else:
        return jaccard

def f1Score(tpos, positives, falseNegative):
    if positives > 0 and tpos+falseNegative > 0:
        precision = tpos/(positives)
        recall = tpos/(tpos+falseNegative)
    else: 
        return 0
    f1Score = 2*(recall*precision)/(recall+precision)
    return f1Score

def main():
    counter = 0
    for img_file in sample:
        detectAndDisplay( img_file, counter )
        counter += 1

main()
