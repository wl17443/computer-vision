import cv2
from matplotlib import pyplot as plt
groundTruth = [[325,60, 157,221],[[55,136,72,77],[239,163,75,80],[367,187,84,70],[504,167,82,81],[636,172,84,83],[31,236,94,94],[179,202,76,85],[284,231,65,87],[414,219,80,97],[554,239,69,81],[676,238,60,78]],[410,113,132,153],[[462,192,95,137],[712,171,130,133]],[] ]
img = cv2.imread('dart5.jpg',0)
img_rgb = cv2.cvtColor(img, cv2.BGR2RGB)
fig, ax = plt.subplots()
print(groundTruth[1])
for face in groundTruth[1]:
    print(face)
    rect1 = plt.Rectangle((face[0], face[1]), face[2],face[3], color='r', fill=False)
    ax.add_artist(rect1)
ax.imshow(img_rgb)
plt.show()
