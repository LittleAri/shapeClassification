"""
Here, we have all the neccesary functions to extract an outline contour from an image.
Note that this code does *not* include functions that could be used for post-contour-extraction
such as smoothing (e.g. with active_contour), or symmetrization (e.g. with SRVF Karcher means).

An example of the contour extraction is as follows:

# 1) Load image:
##################
img = load_resize_image(imgpath,resize=0.75)
# 2) Binarize image:
######################
mpx,mpy,images = binarize(img,white_or_black="both")
# 3) Find outlines contours based on the two binarized versions of the image:
###############################################################################
x1,y1 = get_outline_contour(images[0],0,mpx,mpy,1000)
x2,y2 = get_outline_contour(images[1],0,mpx,mpy,1000)
# 4) Choose least-noisiest contour:
#####################################
newx,newy = choose_contour(x1,y1,x2,y2,reparamTestPoints=500)
# 5) Reparametrize final contour to have n points:
####################################################
xr,yr = reparam(newx,newy,750) 

"""

import numpy as np
from skimage.filters import threshold_otsu
from skimage import measure
from skimage import data, img_as_float, io
from copy import deepcopy
from scipy import ndimage
import os
from skimage import transform
from skimage import color
from PIL import Image
import matplotlib.image as mpimg
from skimage.filters import gaussian
from skimage.segmentation import active_contour


##########################################################################################

##############################
# FUNCTIONS FOR BINARIZATION #
##############################


def binarize(
    img,
    white_or_black="black",
    blackBound=0.2,
    whiteBound=0.15,
    border=20,
    ellipseLeeway=15,
    smallDiamLeeway=0.05,
    largeDiamLeeway=0.45,
    extraFilter=0,
    saveImage=0,
    allImages=0,
):

    try:
        toty, totx = np.shape(img)
    except:
        toty, totx = img.shape

    try:

        # Find the background colour.
        backgroundCol = np.average(img[5, :])

        # Find what value white is.
        mx = 1
        if max([max(img[:, i]) for i in range(0, totx)]) > 100:
            mx = 255  # Since sometimes the greyscale image will load with values between 0-255 instead of 0-1.

        # Find proposed measurements of boundary ellipse.
        mp_x, mp_y, npoint, spoint, wpoint, epoint = ellipseMeasurements(
            img, 0.125, ellipseLeeway, 0.05
        )

        # Find y diamater of ellipse.
        yDiam_smallEllipse = (spoint - npoint) + (toty * 0.05)

        # If y diamter is almost as large as the total image height, then we need a thinner border.
        if yDiam_smallEllipse / toty > 0.76:
            if yDiam_smallEllipse / toty > 0.95:
                border = border + 20
            else:
                border = border + 10

        # Do initial image quantization based on bounds to turn pixels white/black.
        prop, img_edit1 = init_step(
            img, mx, backgroundCol, blackBound, whiteBound, border
        )

        # If at least almost half of the current image is black, then redo the initial edit.
        if prop > 0.45:
            if prop > 0.95:
                mn_b = min(blackBound, whiteBound)
                whiteBound = max(blackBound, whiteBound)
                blackBound = mn_b
            else:
                blackBound = blackBound + 0.1
            prop, img_edit1 = init_step(
                img, mx, backgroundCol, blackBound, whiteBound, border
            )

        if prop > 0.95:
            prop, img_edit1 = init_step(
                img, mx, backgroundCol, blackBound * 2, whiteBound, border
            )

        # Find x and y diamaters for small ellipse and large ellipse.
        xDiam_smallEllipse = (epoint - wpoint) + (totx * smallDiamLeeway)
        yDiam_largeEllipse = (spoint - npoint) + (toty * largeDiamLeeway)
        xDiam_largeEllipse = (epoint - wpoint) + (totx * largeDiamLeeway)

        # Construct ellipsis.
        x_smallEllipse, y_smallEllipse = get_ellipse(
            mp_x, mp_y, xDiam_smallEllipse, yDiam_smallEllipse
        )
        x_largeEllipse, y_largeEllipse = get_ellipse(
            mp_x, mp_y, xDiam_largeEllipse, yDiam_largeEllipse
        )

        img_edit2 = deepcopy(img_edit1)

        # Turn all pixels outside of the large ellipse white.
        yval_range = list(
            range(
                max(int(np.floor(min(y_largeEllipse))), 0),
                min(int(np.ceil(max(y_largeEllipse))) + 1, toty),
            )
        )
        img_edit2[
            : yval_range[0], :
        ] = mx  # All those with y values less than the min of the ellipse are outside of the ellipse.
        img_edit2[
            yval_range[-1] + 1 :, :
        ] = mx  # All those with y values greater than the max of the ellipse are outside of the ellipse.
        for (
            i
        ) in (
            yval_range
        ):  # Check the points that have a y value that falls within the y range of the ellipse.
            vals = np.where(img_edit1[i, :] > -10)[0]
            vals_change = [
                p
                for p in vals
                if in_ellipse(p, i, mp_x, mp_y, xDiam_largeEllipse, yDiam_largeEllipse)
                == 0
            ]
            img_edit2[i, vals_change] = mx

        # Go through each y value within the range of the smaller ellipse.
        # Find the min and max x positions that are black, for each y value. Then turn everything in between black.
        # The aim here is to fill the object black.
        yval_range = list(
            range(
                max(int(np.floor(min(y_smallEllipse))), 0),
                min(int(np.ceil(max(y_smallEllipse))) + 1, toty),
            )
        )
        for i in yval_range:
            vals = np.where(img_edit2[i, :] == 0)[0]
            if len(vals) > 1:
                vals_change = [
                    p
                    for p in vals
                    if in_ellipse(
                        p, i, mp_x, mp_y, xDiam_smallEllipse, yDiam_smallEllipse
                    )
                    == 1
                ]
                if len(vals_change) > 1:
                    img_edit2[i, vals_change[0] : vals_change[-1] + 1] = 0

        # Create a new lower bound and turn everything greater than it to white.
        notWhiteBlack = img_edit2[np.where((img_edit2 != 0) & (img_edit2 != mx))]
        lowerBound = np.average(notWhiteBlack) - np.std(notWhiteBlack)
        img_edit3 = recolour(img_edit2, lowerBound, mx, ">")

        # In this next step we look at the neighbourhood of each pixel within a range. If the majority of the surrounding
        # neighbourhood is white, we turn the pixel, along with a restricted neighbourhood, white.
        notWhiteBlack = np.where((img_edit3 != 0) & (img_edit3 != mx))
        img_edit4 = deepcopy(img_edit3)
        totalRange = len(notWhiteBlack[0])
        for ind in range(0, totalRange):
            i = notWhiteBlack[0][ind]
            j = notWhiteBlack[1][ind]
            vals = img_edit3[i - 10 : i + 10, j - 10 : j + 10].flatten()
            if len(np.where(vals == mx)[0]) >= len(vals) * 0.5:
                img_edit4[i - 2 : i + 2, j - 2 : j + 2] = mx

        if extraFilter == 1:
            # In this extra filter step, everything that is outside of the *small* ellipse that is
            # *not* currently black, will be turned white.
            yval_range = list(
                range(
                    max(int(np.floor(min(y_smallEllipse))), 0),
                    min(int(np.ceil(max(y_smallEllipse))) + 1, toty),
                )
            )
            img_edit5 = deepcopy(img_edit4)
            for (
                i
            ) in (
                yval_range
            ):  # Check the points that have a y value that falls within the y range of the ellipse.
                vals = np.where(img_edit4[i, :] > 0)[0]
                vals_change = [
                    p
                    for p in vals
                    if in_ellipse(
                        p, i, mp_x, mp_y, xDiam_smallEllipse, yDiam_smallEllipse
                    )
                    == 0
                ]
                img_edit5[i, vals_change] = 1
        else:
            img_edit5 = deepcopy(img_edit4)

        # In this final step, we turn all the remaining points to either black or white.
        # If the variable white_or_black == "white", then all points that are *not* black, will be turned white.
        # If the variable white_or_black == "black", then all points that are *not* white, will be turned black.
        # If the variable white_or_black == "both", we provide both the "white" and "black" versions.
        # The "both" option will result in the output being a list of two images.

        if white_or_black == "black":
            finalImage = recolour(img_edit4, mx, 0, "<")
        if white_or_black == "white":
            finalImage = recolour(img_edit4, 0, mx, ">")
        if white_or_black == "both":
            final_image1 = recolour(img_edit4, mx, 0, "<")
            final_image2 = recolour(img_edit4, 0, mx, ">")
            finalImage = [final_image1, final_image2]

        if white_or_black != "both":
            allBlack = len(np.where(finalImage == 0)[0])
            propBlack = allBlack / (totx * toty)
            if propBlack > 0.99:
                finalImage = deepcopy(img)

        if saveImage == 1:
            if white_or_black == "both":
                mpimg.imsave("binimg1.png", final_image1)
                mpimg.imsave("binimg2.png", final_image2)
            else:
                mpimg.imsave("binimg.png", finalImage)

        if allImages == 1:
            finalImage = list(finalImage)
            finalImage.append(img_edit1)
            finalImage.append(img_edit2)
            finalImage.append(img_edit3)
            finalImage.append(img_edit4)

    except:
        finalImage = deepcopy(img)
        mp_x = totx / 2
        mp_y = toty / 2

    proportions = []
    border_y = int(np.floor(np.shape(img)[0] / border))
    border_x = int(np.floor(np.shape(img)[1] / border))
    tot = (np.shape(img)[0] - (border_y * 2)) * (np.shape(img)[1] - (border_x * 2))

    if (allImages == 1) or (white_or_black == "both"):
        for _, img_ in enumerate(finalImage):
            blck = len(np.where(img_ == 0)[0])
            blackProp = blck / tot
            proportions.append(blackProp)
    else:
        blck = len(np.where(finalImage == 0)[0])
        blackProp = blck / tot
        proportions.append(blackProp)

    return mp_x, mp_y, finalImage, proportions


def init_step(img, mx, backgroundCol, blackBound, whiteBound, border):

    img_edit = deepcopy(img)

    eps1 = mx * whiteBound
    eps2 = mx * blackBound

    # Create upper and lower bounds for pixels that should be turned white.
    ub = min([mx, backgroundCol + eps1])
    lb = max([0, backgroundCol - eps1])

    # Create upper and lower bounds for pixels that should be turned black.
    ub2 = min([mx, backgroundCol + eps2])
    lb2 = max([0, backgroundCol - eps2])

    # Find all pixels within the bounds to change white / black.
    w = np.where((img >= lb) & (img <= ub))
    b = np.where((img < lb2) | (img > ub2))

    # Change the colour of those in the bounds.
    img_edit[w] = mx
    img_edit[b] = 0

    # Assuming the object is in the center of the image, change the border pixels so that they're white.
    border_y = int(np.floor(np.shape(img)[0] / border))
    border_x = int(np.floor(np.shape(img)[1] / border))
    img_edit[:border_y, :] = mx
    img_edit[-border_y:, :] = mx
    img_edit[:, :border_x] = mx
    img_edit[:, -border_x:] = mx

    # Calculate proportion of total black pixels.
    tot = (np.shape(img)[0] - (border_y * 2)) * (np.shape(img)[1] - (border_x * 2))
    blck = len(np.where(img_edit == 0)[0])
    prop = blck / tot

    return prop, img_edit


def ellipseMeasurements(img, minCol, leeway, midpointDiff):

    try:
        toty, totx = np.shape(img)
    except:
        toty, totx = img.shape

    # Find darkest pixel.
    eps = (max(img.flatten()) - min(img.flatten())) * minCol
    mn = min(img.flatten()) + eps

    # Find midpoint of image.
    mp1 = int(toty / 2)
    mp2 = int(totx / 2)

    # Find position of nothernmost/southernmost dark pixel, from the center.
    north = min(np.where(img[:mp1, :] < mn)[0])
    south = max(np.where(img[mp1:, :] < mn)[0]) + mp1

    # Find position of easternmost/westernmost dark pixel, from the center.
    west = min(np.where(img[:, :mp2] < mn)[1])
    east = max(np.where(img[:, mp2:] < mn)[1]) + mp2

    # Add leeway to ellipse bounds.
    leeway_x = int(toty / leeway)
    leeway_y = int(totx / leeway)
    npoint = max([north - leeway_x, 10])
    spoint = min([south + leeway_x, toty - 10])
    wpoint = max([west - leeway_y, 10])
    epoint = min([east + leeway_y, totx - 10])

    # Find midpoint of proposed ellipse.
    mp_x = (epoint + wpoint) / 2
    mp_y = (spoint + npoint) / 2

    # Adjust midpoint of ellipse if it's too far from the image midpoint.
    if np.abs(mp_x - mp2) > totx * midpointDiff:
        mp_x = (mp_x + mp2) / 2
    if np.abs(mp_y - mp1) > toty * midpointDiff:
        mp_y = (mp_y + mp1) / 2

    return mp_x, mp_y, npoint, spoint, wpoint, epoint


def get_ellipse(mpx, mpy, diamx, diamy):

    # Denote radius.
    a = diamx / 2  # radius on the x-axis
    b = diamy / 2  # radius on the y-axis

    # Define t.
    t = np.linspace(0, 2 * np.pi, 100)

    # Compute ellipse.
    x = mpx + (a * np.cos(t))
    y = mpy + (b * np.sin(t))

    return x, y


def in_ellipse(xpoint, ypoint, xc, yc, diamx, diamy):

    # Denote radius.
    rx = diamx / 2
    ry = diamy / 2

    # Compute ellipse interval.
    a = ((xpoint - xc) / rx) ** 2
    b = ((ypoint - yc) / ry) ** 2

    # Decide if point is in ellipse.
    inEllipse = 0
    if a + b <= 1:
        inEllipse = 1

    return inEllipse


ops = {">": lambda x, y: x > y, "<": lambda x, y: x < y}


def recolour(img, limit, col, ineq):
    image = deepcopy(img)
    # Ineq(uality) variable should be ">" or "<".

    # Look for pixels less than / greater than some limit.
    pixelsToChange = np.where(ops[ineq](image, limit))

    # Change the colour of the pixels in our bound.
    image[pixelsToChange] = col

    return image


###########################################
# FUNCTIONS FOR CONTOUR FINDING / EDITING #
###########################################


def get_outline_contour(image, mp_x, mp_y, otsu=0, reparamPoints=1000, imgSaved=0):
    # This function extracts the outline contour of an image.
    # Default, otsu = 0, and reparamPoints = 1000.

    try:

        if imgSaved == 0:
            img = image
        else:
            img = io.imread(imgSaved, as_gray=True)

        totx = np.shape(img)[1]
        toty = np.shape(img)[0]

        mx = np.amax(img)

        if len(np.where((img != 0) & (img != mx))[0]) != 0:
            otsu = 1

        if otsu == 1:  # If additional binarization is required.
            thresh = threshold_otsu(img)
            binary = img > thresh
            cont = measure.find_contours(binary, 0.8)
        else:
            cont = measure.find_contours(img, 0.8)

        longest_c = np.argsort([len(c) for c in cont])[
            -10:
        ]  # To find the longest contours

        inds = []

        midpointsx = []
        midpointsy = []

        for ind in longest_c:
            c = cont[ind]
            x = c[:, 1]
            y = c[:, 0]
            rngx = max(x) - min(x)
            rngy = max(y) - min(y)

            if (rngx >= totx / 20) and (rngy >= toty / 20):
                inds.append(ind)
                midpointsx.append((max(x) + min(x)) / 2)
                midpointsy.append((max(y) + min(y)) / 2)

        # We pick the contour that has a midpoint closest to the centre of the image.
        k = np.argmin(
            [
                np.sqrt((mp_x - midpointsx[i]) ** 2 + (mp_y - midpointsy[i]) ** 2)
                for i in range(0, len(midpointsx))
            ]
        )

        k = inds[k]

        xrng = max(cont[k][:, 1]) - min(cont[k][:, 1])
        yrng = max(cont[k][:, 0]) - min(cont[k][:, 0])

        if (xrng < np.shape(img)[1] * 0.05) or (yrng < np.shape(img)[0] * 0.1):
            k = longest_c[-1]

        x_, y_ = reparam(cont[k][:, 1], cont[k][:, 0], reparamPoints)

        mx = 1
        if max([max(img[:, i]) for i in range(0, totx)]) > 100:
            mx = 255  # Since sometimes the greyscale image will load with values between 0-255 instead of 0-1.

        # This step attempts to remove horizontal lines (noise).
        newx, newy = remove_lines(img, x_, y_, mx)

    except:
        print("failed to find contour")
        newx = [0]
        newy = [0]

    return newx, newy


def reparam(x, y, npoints):
    # This function reparametrizes the curve to have npoints.

    tst = np.zeros((len(x), 2))
    tst[:, 0] = x
    tst[:, 1] = y

    p = tst
    dp = np.diff(p, axis=0)
    pts = np.zeros(len(dp) + 1)
    pts[1:] = np.cumsum(np.sqrt(np.sum(dp * dp, axis=1)))
    newpts = np.linspace(0, pts[-1], int(npoints))
    newx = np.interp(newpts, pts, p[:, 0])
    newy = np.interp(newpts, pts, p[:, 1])

    return newx, newy


def div(a, b):
    # This function divides a/b or returns 0 if b == 0.
    if b == 0:
        d = 0
    else:
        d = a / b
    return d


def remove_lines(image, x, y, mx):
    # The aim of this fuction is to remove the points on the curves that form long horizontal lines,
    # which we assume to be noise.

    dx = np.diff(x)
    dy = np.diff(y)

    diffs = [div(dy[i], xp) for i, xp in enumerate(dx)]

    diffs_zero = np.where(np.array(diffs) == 0)[0]
    points_remove = []

    for p in diffs_zero:
        i = int(x[p])
        j = int(y[p])
        l = image[j - 5 : j + 5, i]
        v1 = len(np.where(l == mx)[0]) / len(l)
        l = image[j - 1 : j + 2, i - 3 : i + 3]
        v2 = len(np.where(l == 0)[0]) / len(l)
        if (v1 >= 0.7) and (v2 >= 0.6):
            points_remove.extend(list(range(p - 2, p + 2)))

    x1 = [i for j, i in enumerate(x) if j not in points_remove]
    y1 = [i for j, i in enumerate(y) if j not in points_remove]

    ###########################

    curv = get_curvature(x1, y1)

    avg = np.ceil(np.average(curv))

    points = [ind for ind in range(0, len(x1)) if curv[ind] > avg]

    points2 = []

    for p in points:
        points2.extend([p - 1, p, p + 1])

    newx = [i for j, i in enumerate(x1) if j not in points2]
    newy = [i for j, i in enumerate(y1) if j not in points2]

    return newx, newy


def get_curvature(x, y):

    # This function computes the curvature of a curve.

    dx = np.gradient(x)
    dy = np.gradient(y)

    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    cv = np.abs(dx * d2y - d2x * dy) / ((dx * dx + dy * dy) ** 1.5)

    n = len(cv)

    for p, c in enumerate(cv):
        if np.isnan(c):
            cv[p] = np.inf

    return cv


#########################
# ADDITIONAL FUNCTIONS #
#########################


def load_resize_image(
    imgpath,
    resize=1,
    resizeExtraScale1=0.8,
    resizeExtraScale2=0.55,
    large=1000,
    extraLarge=3000,
):
    # Load original image.
    foo = Image.open((imgpath))
    p = foo.size
    if p[0] > extraLarge:
        resize = resize * resizeExtraScale2
    elif p[0] > large:
        resize = resize * resizeExtraScale1
    # Resize image.
    # This saves on computation cost. If you don't wish to resize, then set resize=1.
    foo = foo.resize((int(p[0] * resize), int(p[1] * resize)), Image.ANTIALIAS)
    # Temporarily save the resized image.
    foo.save("temp.png", quality=100)
    foo.close()
    # Load resized image.
    img = io.imread("temp.png", as_gray=True)
    return img


def choose_contour(x1, y1, x2, y2, reparamTestPoints=500):
    # If the white_or_black parameter is set to "both" in the function 'binarize', then this
    # function can be used to approximate which contour out of the two, is best.
    # The function makes a rough estiamte on the 'best' contour based on which has the
    # smallest sum of curvatures. Note that the function is not perfect, with accuracy ranged from
    # 62% to 100% on test sets containing mussel images. Else, the best 'method' is to check by eye.
    x1_r, y1_r = reparam(x1, y1, reparamTestPoints)
    x2_r, y2_r = reparam(x2, y2, reparamTestPoints)
    c1 = sum(get_curvature(x1_r, y1_r))
    c2 = sum(get_curvature(x2_r, y2_r))
    xr = deepcopy(x1)
    yr = deepcopy(y1)
    if (c1 > c2) and (np.abs(c1 - c2) >= 1):
        xr = deepcopy(x2)
        yr = deepcopy(y2)
    return xr, yr


def snakeSmooth(
    img,
    x,
    y,
    py_or_R="py",
    gauss=True,
    grey=True,
    alpha=0.01,
    beta=3,
    w_line=0,
    w_edge=1,
    gamma=0.01,
    max_px_move=1.0,
    max_iterations=5,
    convergence=0.1,
    boundary_condition="periodic",
    coordinates="rc",
):
    if py_or_R == "py":
        init = np.array([x, y]).T
    else:
        init = np.array([x[0], y[0]]).T
    if gauss == True:
        image = gaussian(img, 3)
    else:
        image = deepcopy(img)
    snake = active_contour(
        image,
        init,
        alpha=alpha,
        beta=beta,
        w_line=w_line,
        w_edge=w_edge,
        gamma=gamma,
        max_px_move=max_px_move,
        max_iterations=max_iterations,
        convergence=convergence,
        boundary_condition=boundary_condition,
        coordinates=coordinates,
    )

    return snake[:, 0], snake[:, 1]
