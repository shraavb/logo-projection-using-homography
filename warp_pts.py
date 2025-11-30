import numpy as np

from est_homography import est_homography

def warp_pts(X, Y, interior_pts):
    """
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts.
        These coordinate describe where a point inside the goal will be warped
        to inside the penn logo. For this assignment, you can keep these new
        coordinates as float numbers.

    """
    # You should Complete est_homography first!
    H = est_homography(X, Y)

    # We want to iterate over each point in X and map it to Y
    # using the homography matrix H for point in X:

    # First we want to make the coordinate to a homogenous coordinate by adding 1
    int_pts_homogeneous = np.hstack([interior_pts, np.ones((interior_pts.shape[0], 1))])

    int_pts_homogeneous = np.array(int_pts_homogeneous)

    # Multiply homography matrix H to the point
    mapped_pts_homogeneous = np.dot(H, int_pts_homogeneous.T).T

    # Next we want to normalize the mapped coordinate to convert it from homogeneous to Cartesian coordinates
    warped_pts = np.zeros((interior_pts.shape[0], 2))

    # We want to divide the result by 3rd term
    for i in range(mapped_pts_homogeneous.shape[0]):
        # Divide by 3rd term
        warped_pts[i, 0] = mapped_pts_homogeneous[i, 0] / mapped_pts_homogeneous[i, 2]  # for x' coords
        warped_pts[i, 1] = mapped_pts_homogeneous[i, 1] / mapped_pts_homogeneous[i, 2]  # for y' coords

    return warped_pts

