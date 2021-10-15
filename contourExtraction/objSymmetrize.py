import numpy as np
from PIL import Image
from skimage import io
import fdasrsf.curve_stats as cs
import pandas as pd
from copy import deepcopy
import fdasrsf.curve_functions as cf
from numpy.linalg import svd
from numpy import (
    zeros,
    sqrt,
    fabs,
    cos,
    sin,
    tile,
    vstack,
    empty,
    cov,
    inf,
    mean,
    arange,
)
from numpy.random import randn
import fdasrsf.utility_functions as uf
import collections
import cv2 as cv
from scipy import interpolate
from scipy.interpolate import interp1d
from joblib import Parallel, delayed


##########################################################################################

##############################
# FUNCTIONS FOR KARCHER MEAN #
##############################


class fdacurve:
    """
    This class provides alignment methods for open and closed curves using the SRVF framework

    Usage: obj = fdacurve(beta, mode, N, scale)
    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param mode: Open ('O') or closed curve ('C') (default 'O')
    :param N: resample curve to N points
    :param scale: scale curve to length 1 (true/false)
    :param q:        (n,T,K) matrix defining n dimensional srvf on T samples with K srvfs
    :param betan:     aligned curves
    :param qn:        aligned srvfs
    :param basis:     calculated basis
    :param beta_mean: karcher mean curve
    :param q_mean:    karcher mean srvf
    :param gams:      warping functions
    :param v:         shooting vectors
    :param C:         karcher covariance
    :param s:         pca singular values
    :param U:         pca singular vectors
    :param coef:      pca coefficients
    :param qun:       cost function
    :param samples:   random samples
    :param gamr:      random warping functions
    :param cent:      center
    :param scale:     scale
    :param E:         energy

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  26-Aug-2020
    """

    def __init__(self, beta1, mode="O", scale=False):
        """
        fdacurve Construct an instance of this class
        :param beta: (n,T,K) matrix defining n dimensional curve on T samples with K curves
        :param mode:  Open ('O') or closed curve ('C') (default 'O')
        :param N: resample curve to N points
        :param scale: scale curve to length 1 (true/false)
        """
        self.mode = mode
        self.scale = scale

        K = beta1.shape[2]
        n = beta1.shape[0]
        N = beta1.shape[1]
        q = zeros((n, N, K))
        for ii in range(0, K):
            a = -cf.calculatecentroid(beta1[:, :, ii])
            beta1[:, :, ii] += tile(a, (N, 1)).T
            try:
                q[:, :, ii] = cf.curve_to_q(beta1[:, :, ii], self.scale, self.mode)
            except:
                q[:, :, ii] = cf.curve_to_q(beta1[:, :, ii], self.mode)[0]

        self.q = q
        self.beta = beta1

    def karcher_mean(self, parallel=False, cores=-1, method="DP"):
        """
        This calculates the mean of a set of curves
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
        """
        n, T, N = self.beta.shape

        modes = ["O", "C"]
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        # Initialize mu as one of the shapes
        mu = self.q[:, :, 0]
        betamean = self.beta[:, :, 0]
        itr = 0

        gamma = zeros((T, N))
        maxit = 20

        sumd = zeros(maxit + 1)
        v = zeros((n, T, N))
        normvbar = zeros(maxit + 1)

        delta = 0.5
        tolv = 1e-4
        told = 5 * 1e-3

        print("Computing Karcher Mean of %d curves in SRVF space.." % N)
        while itr < maxit:
            print("updating step: %d" % (itr + 1))

            if iter == maxit:
                print("maximal number of iterations reached")

            mu = mu / sqrt(cf.innerprod_q2(mu, mu))
            if mode == 1:
                self.basis = cf.find_basis_normal(mu)
            else:
                self.basis = []

            sumv = zeros((n, T))
            sumd[0] = inf
            sumd[itr + 1] = 0
            out = Parallel(n_jobs=cores)(
                delayed(karcher_calc)(
                    self.beta[:, :, n],
                    self.q[:, :, n],
                    betamean,
                    mu,
                    self.basis,
                    mode,
                    method,
                )
                for n in range(N)
            )
            v = zeros((n, T, N))
            for i in range(0, N):
                v[:, :, i] = out[i][0]
                sumd[itr + 1] = sumd[itr + 1] + out[i][1] ** 2

            sumv = v.sum(axis=2)

            # Compute average direction of tangent vectors v_i
            vbar = sumv / float(N)

            normvbar[itr] = sqrt(cf.innerprod_q2(vbar, vbar))
            normv = normvbar[itr]

            if normv > tolv and fabs(sumd[itr + 1] - sumd[itr]) > told:
                # Update mu in direction of vbar
                mu = (
                    cos(delta * normvbar[itr]) * mu
                    + sin(delta * normvbar[itr]) * vbar / normvbar[itr]
                )

                if mode == 1:
                    mu = cf.project_curve(mu)

                x = cf.q_to_curve(mu)
                a = -1 * cf.calculatecentroid(x)
                betamean = x + tile(a, [T, 1]).T
            else:
                break
            itr += 1

        self.q_mean = mu
        self.beta_mean = betamean
        self.v = v
        self.qun = sumd[0 : (itr + 1)]
        self.E = normvbar[0 : (itr + 1)]

        return

    def srvf_align(self, parallel=False, cores=-1, method="DP"):
        """
        This aligns a set of curves to the mean and computes mean if not computed
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
        """
        n, T, N = self.beta.shape

        modes = ["O", "C"]
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        # find mean
        if not hasattr(self, "beta_mean"):
            self.karcher_mean()

        self.qn = zeros((n, T, N))
        self.betan = zeros((n, T, N))
        self.gams = zeros((T, N))
        C = zeros((T, N))
        centroid2 = cf.calculatecentroid(self.beta_mean)
        self.beta_mean = self.beta_mean - tile(centroid2, [T, 1]).T
        q_mu = cf.curve_to_q(self.beta_mean)
        # align to mean

        out = Parallel(n_jobs=-1)(
            delayed(cf.find_rotation_and_seed_coord)(
                self.beta_mean, self.beta[:, :, n], mode, method
            )
            for n in range(N)
        )

        for ii in range(0, N):
            self.gams[:, ii] = out[ii][3]
            self.qn[:, :, ii] = out[ii][1]
            self.betan[:, :, ii] = out[ii][0]

        return


def karcher_calc(beta, q, betamean, mu, basis, closed, method):
    # Compute shooting vector from mu to q_i
    w, d = cf.inverse_exp_coord(betamean, beta, closed, method)

    # Project to tangent space of manifold to obtain v_i
    if closed == 0:
        v = w
    else:
        v = cf.project_tangent(w, q, basis)

    return (v, d)


def compute_kmean(curves, scale=True):
    beta = deepcopy(curves)
    obj_c = fdacurve(beta, mode="O", scale=scale)
    obj_c.srvf_align()
    obj_c.karcher_mean
    km = obj_c.beta_mean
    return list(km[1]), list(km[0])


###################################
# FUNCTIONS FOR CONTOUR SPLITTING #
###################################


def get_contour_side(X, Y):
    # This function finds the left and right side of the outline contour.
    # Input: outline contour and D, i.e. desired side or shortest side.
    # Output: left and right side of an outline contour.

    try:
        p = 0

        # 1) Find centre of pot.
        #########################
        xc, yc, coords = find_centre(X, Y)

        # 2) Define centre range.
        #########################
        # Find all points that are close to the centre. We hope these will be the points on the top and the base of the pot.
        tnth = min(X) + (max(X) - min(X)) / 10
        lb = xc - tnth
        ub = xc + tnth
        rng = []

        rng = np.where((X >= lb) & (X <= ub))[0]

        # 3) Define boundary of body of pot
        ###################################
        lb_y = min(Y) + ((max(Y) - min(Y)) / 3)
        ub_y = min(Y) + (2 * (max(Y) - min(Y)) / 3)

        # 4) Find top and bottom and bot.
        #################################
        ktop = np.argmax(Y[rng])
        top_pnt = rng[ktop]

        kbot = np.argmin(Y[rng])
        bot_pnt = rng[kbot]

        mx_pnt = max(top_pnt, bot_pnt)
        mn_pnt = min(top_pnt, bot_pnt)

        # We check if the correct top and bottom of the pot have been found.
        if np.abs(Y[mx_pnt] - Y[mn_pnt]) < (np.max(Y) - np.min(Y)) / 5:
            # If the difference between the top and bottom of the pot is too small, then define new top and bottom points.
            mn_pnt = np.argmin(Y)
            mx_pnt = np.argmax(Y)

        # 5) Define the left and right side of the outline contour.
        ###########################################################
        xs1 = list(X[mn_pnt:mx_pnt])
        ys1 = list(Y[mn_pnt:mx_pnt])

        xs2 = list(X)[mx_pnt:] + list(X)[:mn_pnt]
        ys2 = list(Y)[mx_pnt:] + list(Y)[:mn_pnt]
    except:
        mn_pnt = np.argmin(Y)
        mx_pnt = np.argmax(Y)

        mx = max(mx_pnt, mn_pnt)
        mn = min(mx_pnt, mn_pnt)

        xs1 = list(X[mn:mx])
        ys1 = list(Y[mn:mx])
        xs2 = list(X[mx - 1 :]) + list(X[:mn])
        ys2 = list(Y[mx - 1 :]) + list(Y[:mn])

    if (len(xs1) == 0) or (len(xs2) == 0):
        mn_pnt = np.argmin(Y)
        mx_pnt = np.argmax(Y)

        mx = max(mx_pnt, mn_pnt)
        mn = min(mx_pnt, mn_pnt)

        xs1 = list(X[mn:mx])
        ys1 = list(Y[mn:mx])
        xs2 = list(X[mx - 1 :]) + list(X[:mn])
        ys2 = list(Y[mx - 1 :]) + list(Y[:mn])

    return xs1, ys1, xs2, ys2


def find_centre(x, y):
    # Finds coordinates for the centre of the pot, by focusing on the middle of the pot.
    # Input: outline contour
    # Output: co-ordinates for the centre of pot, and the coordinates of the middle points.

    # 1) Decide boundaries
    ######################
    yc = min(y) + (max(y) - min(y)) / 2
    tnth = max(y) - (max(y) - min(y)) / 10
    coords = np.where((np.array(y) < tnth) & (np.array(y) > yc))[0]

    # 2) Find centre
    #################
    xc = (
        min(np.array(x)[coords])
        + (max(np.array(x)[coords]) - min(np.array(x)[coords])) / 2
    )

    return xc, yc, coords


def edit_ends(x1, y1, x2, y2, xc):
    # This functions tries to ensure that both sides of the contour start/end at the centre of the vase, and
    # also classifies the sides as 'left' and 'right'.
    # Input: the x and y coordinates of both sides of the contour.
    # Output: the left side of the contour, and right side of the contour.

    if y1[0] < y1[-1]:
        x1.reverse()
        y1.reverse()
    if y2[0] < y2[-1]:
        x2.reverse()
        y2.reverse()

    if min(x1) < min(x2):
        xL = deepcopy(x1)
        yL = deepcopy(y1)
        xR = deepcopy(x2)
        yR = deepcopy(y2)
    else:
        xR = deepcopy(x1)
        yR = deepcopy(y1)
        xL = deepcopy(x2)
        yL = deepcopy(y2)

    new_Lx = []
    new_Ly = []

    new_Lx.append(xc)
    new_Ly.append(yL[0])

    new_Lx.extend(list(np.array(list(xL))[list(np.where(xL < xc)[0])]))
    new_Ly.extend(list(np.array(list(yL))[list(np.where(xL < xc)[0])]))

    new_Lx.append(xc)
    new_Ly.append(yL[-1])

    new_Rx = []
    new_Ry = []

    new_Rx.append(xc)
    new_Ry.append(yR[0])

    new_Rx.extend(list(np.array(list(xR))[list(np.where(xR > xc)[0])]))
    new_Ry.extend(list(np.array(list(yR))[list(np.where(xR > xc)[0])]))

    new_Rx.append(xc)
    new_Ry.append(yR[-1])

    return new_Lx, new_Ly, new_Rx, new_Ry


###########################################
# ADDITIONAL FUNCTIONS FOR SYMMETRIZATION #
###########################################


def cut_ends(x, y, proportion=10):
    # The aim of this function is to cater for photography-angle issues. This is used if the inside of the mouth
    # of a vase is showing and if the base is more curves than straight. The function creates straight, horizontal
    # cuts and the top and bottom of the vase to cater for this.
    # Input: x and y coordinates of vase contour.
    # Output: updated x and y coordinates with the top and bases cut.

    eps = (max(y) - min(y)) / proportion
    yub = max(y) - eps
    ylb = min(y) + eps
    rng = np.where(np.array(y) >= yub)[0]
    xub_L = np.argmin(np.array(x)[rng])
    xub_L = rng[xub_L]
    xub_R = np.argmax(np.array(x)[rng])
    xub_R = rng[xub_R]
    rng = np.where(np.array(y) <= ylb)[0]
    xlb_L = np.argmin(np.array(x)[rng])
    xlb_L = rng[xlb_L]
    xlb_R = np.argmax(np.array(x)[rng])
    xlb_R = rng[xlb_R]
    cuts = [xub_L, xub_R, xlb_R, xlb_L]
    mins = np.argsort(cuts)
    # Work out whether you're going to start with a cut.
    if ((mins[0] < 2) and (mins[1] < 2)) or ((mins[0] >= 2) and (mins[1] >= 2)):
        initcut = 1
    else:
        initcut = 0

    new_x = []
    new_y = []

    if initcut == 0:
        # No cut at the start.
        new_x.extend(x[cuts[mins[0]] : cuts[mins[1]] + 1])
        new_y.extend(y[cuts[mins[0]] : cuts[mins[1]] + 1])
        # Make first cut.
        new_x.extend(x[cuts[mins[2]] : cuts[mins[3]] + 1])
        new_y.extend(y[cuts[mins[2]] : cuts[mins[3]] + 1])
        # Make second cut.
        new_x.append(new_x[0])
        new_y.append(new_y[0])
    else:
        # Start with cut.
        new_x.extend(x[cuts[mins[1]] : cuts[mins[2]] + 1])
        new_y.extend(y[cuts[mins[1]] : cuts[mins[2]] + 1])
        # Make second cut.
        new_x.extend(x[cuts[mins[3]] :])
        new_y.extend(y[cuts[mins[3]] :])
        new_x.extend(x[: cuts[mins[0]] + 1])
        new_y.extend(y[: cuts[mins[0]] + 1])
        new_x.append(new_x[0])
        new_y.append(new_y[0])

    return new_x, new_y


def reparam(x, y, npoints=300):
    # This function reparametrizes the functions to have n points.
    # Input: x and y coordinates of contour and the desired number of points.
    # Output: reparametrized x and y values.

    tst = np.zeros((len(x), 2))
    tst[:, 0] = x
    tst[:, 1] = y

    p = tst
    dp = np.diff(p, axis=0)
    pts = np.zeros(len(dp) + 1)
    pts[1:] = np.cumsum(np.sqrt(np.sum(dp * dp, axis=1)))
    newpts = np.linspace(0, pts[-1], npoints)
    newx = np.interp(newpts, pts, p[:, 0])
    newy = np.interp(newpts, pts, p[:, 1])

    return newx, newy


def rescale(x, y, scaledHeightWidth=3, scaleDirection="both"):
    # This function rescales the contour so that it's between certain values (of a certain width/height)
    # and centred at (0,0). The 'scaleDirection' parameter is set to "both" as defult. This means that
    # it scales the max(height,width) to the desired width/height. The smaller side (heigh/width) will
    # then be scaled with the same scale factor, and hence lie within the desired range (and not greater).
    # If it's desired that the height will be of a certain length, or the width, then set the
    # scaleDirection paramter to "height" or "width" respectively.
    # Input: x, y coordinates, desired max height/width, and desired direction (height/width/max)
    # Output: rescaled x and y coordinates.

    if type(x) != np.ndarray:
        x = np.array(x)

    if type(y) != np.ndarray:
        y = np.array(y)

    if scaleDirection == "height":
        p = max(y) - min(y)
    elif scaleDirection == "width":
        p = max(x) - min(x)
    else:
        p = max(max(y) - min(y), max(x) - min(x))

    q = p / scaledHeightWidth

    x_ = x / q
    y_ = y / q
    mp = min(x_) + (max(x_) - min(x_)) / 2
    x_ = x_ - mp
    mp = min(y_) + (max(y_) - min(y_)) / 2
    y_ = y_ - mp

    return x_, y_


def symmetrize(
    x,
    y,
    sideReparamPoints=200,
    rep=0,
    KarcherMeanScale=True,
    cutEnds=True,
    reparamPoints=250,
    scaledHeightWidth=3,
    scaleDirection="both",
    proportion=10,
):
    # This function symmetrizes a vase contour by taking the x,y coordinates, splitting it half
    # down the centre, computing the Karcher mean of the two vases, and then joining the
    # Karcher mean to its reflection in order to create a symmetrized closed contour.
    # Input: original x and y coordinates of the closed contour.
    # Output: new x and y coordinates of a symmetric contour.

    # 1) Rescale smoothed contour so that the height is 1.
    x_s = x / (max(y) - min(y))
    y_s = y / (max(y) - min(y))

    # 2) Find m=max(x)-min(x)
    m = max(x_s) - min(x_s)

    # 3) Find centrpoint of the top.
    ub = max(y_s) - ((max(y_s) - min(y_s)) / 6)
    x_ = x_s[np.where(y_s > ub)[0]]
    xcu = max(x_) - ((max(x_) - min(x_)) / 2)

    # 4) Get two sides of smoothed rescaled contour and edit the sides.
    xL1, yL1, xR1, yR1 = get_contour_side(x_s, y_s)
    x_L, y_L, x_R, y_R = edit_ends(xL1, yL1, xR1, yR1, xcu)

    # 5) Reparametrize both sides.
    xLr, yLr = reparam(x_L, y_L, sideReparamPoints)
    xRr, yRr = reparam(x_R, y_R, sideReparamPoints)

    # 6) Reflect the left side and make both (x) sides start from 0.
    xLr = -1 * xLr
    xLr = xLr - min(xLr)
    xRr = xRr - min(xRr)

    # 7) Save curves into one array.
    if rep == 1:
        xRr = deepcopy(xLr)
        yRr = deepcopy(yLr)
    if rep == 2:
        xLr = deepcopy(xRr)
        yLr = deepcopy(yRr)

    curves = np.zeros((2, sideReparamPoints, 2))
    curves[0, :, 0] = yLr
    curves[1, :, 0] = xLr
    curves[0, :, 1] = yRr
    curves[1, :, 1] = xRr

    # 8) Karcher mean on the two sides.
    xm, ym = compute_kmean(curves, scale=KarcherMeanScale)

    if ym[0] < ym[-1]:
        xm = xm[::-1]
        ym = ym[::-1]

    # 9) Rescale Karcher mean so that the height is 1. (xm,ym)
    xm_s = xm / (max(ym) - min(ym))
    ym_s = ym / (max(ym) - min(ym))

    # 10) Edit the sides of KM so it starts and ends at the same point.
    coords = np.where(xm_s >= min(xm_s))[0]
    xm_s = xm_s[coords]
    ym_s = ym_s[coords]

    # 11) Reflect the Karcher mean and edit its ends.
    xm_L = (-1 * xm_s) + (max(xm_s) - min(-1 * (xm_s))) - m
    coords = np.where(xm_L < min(xm_s))[0]
    xm_L = xm_L[coords]
    ym_L = ym_s[coords]

    # 12) Put everything together.
    new_x = []
    new_y = []
    new_x.extend(list(xm_s))
    new_y.extend(list(ym_s))
    new_x.extend(list(xm_L[::-1]))
    new_y.extend(list(ym_L[::-1]))

    if new_x[-1] != new_x[0]:
        new_x.append(new_x[0])
        new_y.append(new_y[0])

    # 13) Rescale and reparam new coords so that they have 139 points.
    if cutEnds == True:
        new_x, new_y = cut_ends(new_x, new_y, proportion=proportion)
    new_xs, new_ys = rescale(
        np.array(new_x),
        np.array(new_y),
        scaledHeightWidth=scaledHeightWidth,
        scaleDirection=scaleDirection,
    )
    final_x, final_y = reparam(new_xs, new_ys, npoints=reparamPoints)

    inds = np.argsort(final_y)[-3:]
    mn = min(min(final_y[inds]), scaledHeightWidth / 2)
    final_y[inds] = mn

    return final_x, final_y


def reorderPoints(
    x, y, rotate=True, reorder=True, scaledHeightWidth=3, direction="clockwise"
):
    # 1) Rotate the object by 180 degrees, if desired.
    if rotate == True:
        x, y = rescale(x, -1 * y)
    # 2) Make the points go in the desired direction. Default is "clockwise".
    mp = min(x) + (max(x) - min(x)) / 2
    rng = np.where(np.array(y) > 1)[0]
    k = np.argmin(abs(np.array(x)[rng] - mp))
    k = rng[k]
    try:
        if (x[k] > x[k + 2]) and (direction == "clockwise"):
            x = x[::-1]
            y = y[::-1]
    except:
        if (x[k - 2] > x[k]) and (direction == "clockwise"):
            x = x[::-1]
            y = y[::-1]
    # 3) Reorder the points so that they start from the top centre.
    if reorder == True:
        eps = scaledHeightWidth / 10
        rng = np.where(np.array(y) > max(y) - eps)[0]
        p = np.argsort(abs(x[rng]))[0]
        p = rng[p]
        newx = []
        newy = []
        newx.extend(x[p:])
        newy.extend(y[p:])
        newx.extend(x[:p])
        newy.extend(y[:p])
        newx[-1] = newx[0]
    else:
        newx = deepcopy(x)
        newy = deepcopy(y)

    return newx, newy


def procrustes(F1, F2, scaling=True, rotation=True, reflection=True):
    # Procrustes alignment

    # Translate
    muF1 = F1.mean(axis=0)
    muF2 = F2.mean(axis=0)

    F1_0 = F1 - muF1
    F2_0 = F2 - muF2

    n, m = F1.shape
    ny, my = F2.shape

    # Scaling
    sF1 = np.sum(F1_0 ** 2)
    sF2 = np.sum(F2_0 ** 2)

    F1_0 /= np.sqrt(sF1)
    F2_0 /= np.sqrt(sF2)

    # Rotation
    P = np.dot(F1_0.T, F2_0)
    U, s, V = np.linalg.svd(P)
    V = V.T
    R = np.dot(V, U.T)

    # Reflection
    if not reflection:
        if np.linalg.det(V.T) < 0:
            V[:, -1] = -V[:, -1]
            s[-1] = -s[:1]
            R = np.dot(V, U.T)

    if not rotation:
        R = np.eye(np.shape(R)[0])

    if scaling:
        b = np.sum(s) * np.sqrt(sF1) / np.sqrt(sF2)
        # residual
        r = 1 - np.sum(s) ** 2
        # transformed coords
        Z = np.sqrt(sF1) * np.sum(s) * np.dot(F2_0, R) + muF1
    else:
        b = 1
        # residual
        r = 1 + sF2 / sF1 - 2 * np.sum(s) * np.sqrt(sY) / np.sqrt(sF1)
        # transformed coords
        Z = np.sqrt(sF2) * np.dot(F2_0, R) + muF1

    # return residual, transformed coords, rotation, scale, translation
    return r, Z, R, b, muF1 - b * np.dot(muF2, R)
