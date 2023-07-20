import numpy as np
import cv2
import time

from numpy import ndarray

image = cv2.imread('y.png', cv2.IMREAD_UNCHANGED)

height, width, channels = image.shape
img_copy = image
pixelweight = np.zeros((height, width), dtype=np.uint32)
pixelgroup: ndarray = np.zeros((height, width), dtype=np.uint32)
lookedpixels = np.zeros((height, width), dtype=np.bool_)
locallookedpixels = np.zeros((height, width), dtype=np.bool_)
clusters = []
groupnumber = 1
neighgroups = set()
getpixelsofgroups = []


def simplify_image(similarity_threshold):
    for h in range(0, height):
        for w in range(0, width):
            pixelWeightCalculator(h, w)
            print("finding pixel weights", h, w)

    for h in range(0, height):
        for w in range(0, width):
            if channels == 3 or (channels == 4 and image[h, w, 3] != 0):
                #print("finding neighborhoods", h, w)
                shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                for dx, dy in shifts:
                    nx, ny = h + dx, w + dy
                    if (nx == h and ny == w) or not 0 <= nx < height or not 0 <= ny < width or pixelgroup[nx, ny] == 0:
                        continue
                    if pixelgroup[h, w] != pixelgroup[nx, ny] and (channels == 3 or (channels == 4 and image[nx, ny, 3] != 0)) and differ(image[h, w][:3], image[nx, ny][:3]) < similarity_threshold and not (pixelgroup[nx, ny], pixelgroup[h, w]) in neighgroups:
                        neighgroups.add((pixelgroup[h, w], pixelgroup[nx, ny]))

    priority = []
    for sort in neighgroups:
        print("sorting data")
        x, y = sort
        indice1 = getpixelsofgroups[x - 1]
        indice2 = getpixelsofgroups[y - 1]
        indice1color = img_copy[indice1[0]][:3]
        indice2color = img_copy[indice2[0]][:3]
        calc = 100 / (differ(indice1color, indice2color))
        if len(indice1) > len(indice2):
            #calc = calc * (len(indice1) / len(indice2))
            priority.append(calc)
        elif len(indice1) < len(indice2):
            #calc = calc * (len(indice2) / len(indice1))
            priority.append(calc)

    mylist = sort_by_second_list(neighgroups, priority)
    for m in range(len(mylist)):
        print("simplifying", m, len(mylist))
        x, y = mylist[m]
        if x == y:
            continue
        indice1 = getpixelsofgroups[x - 1]
        indice2 = getpixelsofgroups[y - 1]
        indice1color = img_copy[indice1[0]]
        indice2color = img_copy[indice2[0]]
        avgcolor = [int(round((indice1color[i] * len(indice1) + indice2color[i] * len(indice2)) / (len(indice1) + len(indice2)))) for i in range(4)]
        if len(indice1) > len(indice2):
            for i in range(0, len(indice2)):
                img_copy[indice2[i]] = avgcolor
            for i in range(0, len(indice1)):
                img_copy[indice1[i]] = avgcolor
            mylist = [list(map(lambda item: x if item == y else item, tup)) if y in tup else tup for tup in mylist]
            pixelgroup[pixelgroup == y] = x
            getpixelsofgroups[x - 1] += (getpixelsofgroups[y - 1])
        elif len(indice2) > len(indice1):
            for i in range(0, len(indice1)):
                img_copy[indice1[i]] = avgcolor
            for i in range(0, len(indice2)):
                img_copy[indice2[i]] = avgcolor
            mylist = [list(map(lambda item: y if item == x else item, tup)) if x in tup else tup for tup in mylist]
            pixelgroup[pixelgroup == x] = y
            getpixelsofgroups[y - 1] += (getpixelsofgroups[x - 1])
        else:
            color = mean(indice1color, indice2color)
            for i in range(0, len(indice1)):
                img_copy[indice1[i]] = color
            for i in range(0, len(indice2)):
                img_copy[indice2[i]] = color
            mylist = [list(map(lambda item: x if item == y else item, tup)) if y in tup else tup for tup in mylist]
        mylist = [list(tup) for tup in mylist]


def sort_by_second_list(first_list, second_list):
    combined_list = list(zip(list(first_list), second_list))
    sorted_list = sorted(combined_list, key=lambda x: -x[1])
    return [x[0] for x in sorted_list]


def differ(first, second):
    first = np.array(first, dtype=np.uint32)
    second = np.array(second, dtype=np.uint32)
    return np.sum((first - second)**2)


def colorsaresame(first, second):
    first = np.array(first, dtype=np.int16)
    second = np.array(second, dtype=np.int16)
    for i in range(0, len(first)):
        if first[i] - second[i] != 0:
            return False
    return True


def mean(first, second):
    first = np.array(first, dtype=np.int16)
    second = np.array(second, dtype=np.int16)
    return [(first[0] + second[0])/2, (first[1] + second[1])/2, (first[2] + second[2])/2, (first[3] + second[3])/2]


def pixelWeightCalculator(h, w):
    global groupnumber
    if pixelweight[h, w] == 0 and (channels == 3 or (channels == 4 and image[h, w, 3] != 0)):
        findClusters(h, w)
        locallookedpixels.fill(False)
        for (x, y) in clusters:
            pixelweight[x, y] = len(clusters)
            pixelgroup[x, y] = groupnumber
        getpixelsofgroups.append(list(clusters))
        clusters.clear()
        groupnumber = groupnumber + 1
    return pixelweight[h, w]


def findClusters(i, j):
    clusters.append((i, j))
    locallookedpixels[i, j] = True

    lefttopstack = []
    leftstack = []
    leftbottomstack = []
    topstack = []
    bottomsstack = []
    righttopsstack = []
    rightsstack = []
    rightbottomsstack = []

    top = max(i - 1, 0)
    bottom = min(i + 1, height - 1)
    right = min(j + 1, width - 1)
    left = max(j - 1, 0)

    lefttopstack.append((top, left))
    righttopsstack.append((top, right))
    topstack.append((top, j))
    bottomsstack.append((bottom, j))
    rightbottomsstack.append((bottom, right))
    rightsstack.append((i, right))
    leftstack.append((i, left))
    leftbottomstack.append((bottom, left))

    while bottomsstack or rightsstack or rightbottomsstack or leftbottomstack or leftstack or topstack or lefttopstack or righttopsstack:
        while topstack:
            x, y = topstack.pop()
            if not locallookedpixels[x, y]:
                locallookedpixels[x, y] = True
                if colorsaresame(image[i, j], image[x, y]):
                    clusters.append((x, y))
                    n = max(x - 1, 0)
                    m = min(x + 1, height - 1)
                    k = min(y + 1, width - 1)
                    l = max(y - 1, 0)

                    if x != 0:
                        topstack.append((n, y))
                        if y == 0:
                            righttopsstack.append((n, k))
                            rightbottomsstack.append((m, k))
                            rightsstack.append((x, k))
                            continue
                        elif y == width - 1:
                            lefttopstack.append((n, l))
                            leftstack.append((x, l))
                            leftbottomstack.append((m, l))
                            continue
                        else:
                            lefttopstack.append((n, l))
                            righttopsstack.append((n, k))
                            rightbottomsstack.append((m, k))
                            rightsstack.append((x, k))
                            leftstack.append((x, l))
                            leftbottomstack.append((m, l))
                            continue
                    else:
                        if y == 0:
                            rightbottomsstack.append((m, k))
                            rightsstack.append((x, k))
                            continue
                        elif y == width - 1:
                            leftstack.append((x, l))
                            leftbottomstack.append((m, l))
                            continue
                        else:
                            rightbottomsstack.append((m, k))
                            rightsstack.append((x, k))
                            leftstack.append((x, l))
                            leftbottomstack.append((m, l))
        while bottomsstack:
            x, y = bottomsstack.pop()
            if not locallookedpixels[x, y]:
                locallookedpixels[x, y] = True
                if colorsaresame(image[i, j], image[x, y]):
                    clusters.append((x, y))
                    n = max(x - 1, 0)
                    m = min(x + 1, height - 1)
                    k = min(y + 1, width - 1)
                    l = max(y - 1, 0)

                    if x != height - 1:
                        bottomsstack.append((m, y))
                        if y == 0:
                            righttopsstack.append((n, k))
                            rightbottomsstack.append((m, k))
                            rightsstack.append((x, k))
                            continue
                        elif y == width - 1:
                            lefttopstack.append((n, l))
                            leftstack.append((x, l))
                            leftbottomstack.append((m, l))
                            continue
                        else:
                            lefttopstack.append((n, l))
                            righttopsstack.append((n, k))
                            rightbottomsstack.append((m, k))
                            rightsstack.append((x, k))
                            leftstack.append((x, l))
                            leftbottomstack.append((m, l))
                            continue
                    else:
                        if y == 0:
                            righttopsstack.append((n, k))
                            rightsstack.append((x, k))
                            continue
                        elif y == width - 1:
                            lefttopstack.append((n, l))
                            leftstack.append((x, l))
                            continue
                        else:
                            lefttopstack.append((n, l))
                            righttopsstack.append((n, k))
                            rightsstack.append((x, k))
                            leftstack.append((x, l))
        while rightsstack:
            x, y = rightsstack.pop()
            if not locallookedpixels[x, y]:
                locallookedpixels[x, y] = True
                if colorsaresame(image[i, j], image[x, y]):
                    clusters.append((x, y))
                    n = max(x - 1, 0)
                    m = min(x + 1, height - 1)
                    k = min(y + 1, width - 1)
                    l = max(y - 1, 0)

                    if y != width - 1:
                        rightsstack.append((x, k))
                        if x == 0:
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
                            leftbottomstack.append((m, l))
                            continue
                        elif x == height - 1:
                            lefttopstack.append((n, l))
                            righttopsstack.append((n, k))
                            topstack.append((n, y))
                            continue
                        else:
                            lefttopstack.append((n, l))
                            righttopsstack.append((n, k))
                            topstack.append((n, y))
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
                            leftbottomstack.append((m, l))
                            continue
                    else:
                        if x == 0:
                            bottomsstack.append((m, y))
                            leftbottomstack.append((m, l))
                            continue
                        elif x == height - 1:
                            lefttopstack.append((n, l))
                            topstack.append((n, y))
                            continue
                        else:
                            lefttopstack.append((n, l))
                            topstack.append((n, y))
                            bottomsstack.append((m, y))
                            leftbottomstack.append((m, l))
        while leftstack:
            x, y = leftstack.pop()
            if not locallookedpixels[x, y]:
                locallookedpixels[x, y] = True
                if colorsaresame(image[i, j], image[x, y]):
                    clusters.append((x, y))
                    n = max(x - 1, 0)
                    m = min(x + 1, height - 1)
                    k = min(y + 1, width - 1)
                    l = max(y - 1, 0)

                    if y != 0:
                        leftstack.append((x, l))
                        if x == 0:
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
                            leftbottomstack.append((m, l))
                            continue
                        elif x == height - 1:
                            lefttopstack.append((n, l))
                            righttopsstack.append((n, k))
                            topstack.append((n, y))
                            continue
                        else:
                            lefttopstack.append((n, l))
                            righttopsstack.append((n, k))
                            topstack.append((n, y))
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
                            leftbottomstack.append((m, l))
                            continue
                    else:
                        if x == 0:
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
                            continue
                        elif x == height - 1:
                            righttopsstack.append((n, k))
                            topstack.append((n, y))
                            continue
                        else:
                            righttopsstack.append((n, k))
                            topstack.append((n, y))
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
        while lefttopstack:
            x, y = lefttopstack.pop()
            if not locallookedpixels[x, y]:
                locallookedpixels[x, y] = True
                if colorsaresame(image[i, j], image[x, y]):
                    clusters.append((x, y))
                    n = max(x - 1, 0)
                    m = min(x + 1, height - 1)
                    k = min(y + 1, width - 1)
                    l = max(y - 1, 0)

                    bottomsstack.append((m, y))
                    rightsstack.append((x, k))
                    if y != 0:
                        leftstack.append((x, l))
                        leftbottomstack.append((m, l))
                        if x != 0:
                            lefttopstack.append((n, l))
                            righttopsstack.append((n, k))
                            topstack.append((n, y))
                            continue
                        continue
                    else:
                        if x != 0:
                            righttopsstack.append((n, k))
                            topstack.append((n, y))
        while righttopsstack:
            x, y = righttopsstack.pop()
            if not locallookedpixels[x, y]:
                locallookedpixels[x, y] = True
                if colorsaresame(image[i, j], image[x, y]):
                    clusters.append((x, y))
                    n = max(x - 1, 0)
                    m = min(x + 1, height - 1)
                    k = min(y + 1, width - 1)
                    l = max(y - 1, 0)

                    bottomsstack.append((m, y))
                    leftstack.append((x, l))
                    if y != width - 1:
                        rightsstack.append((x, k))
                        rightbottomsstack.append((m, k))
                        if x != height - 1:
                            topstack.append((n, y))
                            righttopsstack.append((n, k))
                            lefttopstack.append((n, l))
                            continue
                        continue
                    else:
                        if x != height - 1:
                            topstack.append((n, y))
                            lefttopstack.append((n, l))
        while leftbottomstack:
            x, y = leftbottomstack.pop()
            if not locallookedpixels[x, y]:
                locallookedpixels[x, y] = True
                if colorsaresame(image[i, j], image[x, y]):
                    clusters.append((x, y))
                    n = max(x - 1, 0)
                    m = min(x + 1, height - 1)
                    k = min(y + 1, width - 1)
                    l = max(y - 1, 0)

                    topstack.append((n, y))
                    rightsstack.append((x, k))
                    if y != 0:
                        leftstack.append((x, l))
                        lefttopstack.append((n, l))
                        if x != height - 1:
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
                            leftbottomstack.append((m, l))
                            continue
                        continue
                    else:
                        if x != height - 1:
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
        while rightbottomsstack:
            x, y = rightbottomsstack.pop()
            if not locallookedpixels[x, y]:
                locallookedpixels[x, y] = True
                if colorsaresame(image[i, j], image[x, y]):
                    clusters.append((x, y))
                    n = max(x - 1, 0)
                    m = min(x + 1, height - 1)
                    k = min(y + 1, width - 1)
                    l = max(y - 1, 0)

                    topstack.append((n, y))
                    leftstack.append((x, l))
                    if y != width - 1:
                        rightsstack.append((x, k))
                        righttopsstack.append((n, k))
                        if x != height - 1:
                            bottomsstack.append((m, y))
                            rightbottomsstack.append((m, k))
                            leftbottomstack.append((m, l))
                            continue
                        continue
                    else:
                        if x != height - 1:
                            bottomsstack.append((m, y))
                            leftbottomstack.append((m, l))

simplify_image(1000)
cv2.imwrite('y.png', img_copy)
cv2.waitKey(0)

