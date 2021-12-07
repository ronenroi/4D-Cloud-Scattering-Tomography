import os
import sys
# import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom


def StringOfPearls(SATS_NUMBER=10, orbit_altitude=500):
    """
    Set orbit parmeters:
         returns sat_positions: np.array of shape (SATS_NUMBER,3).
         The satellites setup alwas looks like \ \ | / /.
    """
    r_earth = 6371.0  # km
    Rsat = orbit_altitude  # km orbit altitude
    R = r_earth + Rsat
    r_orbit = R
    Darc = 73.5# 100  # km # distance between adjecent satellites (on arc).
    Dtheta = Darc / R  # from the center of the earth.


    # where to set the sateliites?
    theta_config = np.arange(-0.5 * SATS_NUMBER, 0.5 * SATS_NUMBER) * Dtheta
    theta_config = theta_config[::-1]  # put sat1 to be the rigthest
    X_config = r_orbit * np.sin(theta_config)
    Z_config = r_orbit * np.cos(theta_config) - r_earth
    Y_config = np.zeros_like(X_config)
    sat_positions = np.vstack([X_config, Y_config, Z_config])  # path.shape = (3,#sats) in km.

    return sat_positions.T


class SpaceMultiView(shdom.MultiViewProjection):
    """
    Inherent from pyshdom MultiViewProjection. We will use the Perspective projection"""

    def __init__(self):
        super().__init__()
        self._sat_positions = None  # it will be set by set_satellites() and it is np.array.
        self._lookat_list = None
        self._fov_list = None
        self._nx_list = None
        self._ny_list = None
        self._up_list = None

    def set_satellites(self, sat_positions, sat_lookats):
        """
        Parameters:
        input:
         - sat_positions: x,y,z of the satellites (each row has x,y,z of some sat).
           np.array shape of (#sats,3)
         - similar for the sat_lookats, it is where the satellites look at.
           np.array shape of (#sats,3)

        """
        self._sat_positions = sat_positions
        self._lookat_list = sat_lookats

        self._up_list = np.array(len(sat_positions) * [0, 1, 0]).reshape(-1, 3)

    def set_commonCameraIntrinsics(self, FOV=0.1, nx=64, ny=64):
        assert self._sat_positions is not None, "Before using it, use upper set method"

        self._fov_list = np.tile(FOV, len(self._sat_positions))
        self._nx_list = np.tile(nx, len(self._sat_positions))
        self._ny_list = np.tile(ny, len(self._sat_positions))

    def update_views(self, names=None):

        # nameas:
        if names is None:
            names = ["sat" + str(i + 1) for i in range(len(self._sat_positions))]

        for pos, lookat, fov, nx, ny, up, name in zip(self._sat_positions, \
                                                      self._lookat_list, \
                                                      self._fov_list, \
                                                      self._nx_list, \
                                                      self._ny_list, \
                                                      self._up_list, names):
            x, y, z = pos
            loop_projection = shdom.PerspectiveProjection(fov, nx, ny, x, y, z)
            loop_projection.look_at_transform(lookat, up)
            self.add_projection(loop_projection, name)

    def show_setup(self, scale=0.6, axisWidth=3.0, axisLenght=1.0, FullCone=False):
        """
        Show all the views:
        """
        for view, view_name in zip(self._projection_list, self._names):
            view.show_camera(scale, axisWidth, axisLenght, FullCone)

            t = view.position
            mlab.text3d(t[0] + 0.2 * scale, t[1], t[2] + 0.2 * scale, view_name, color=(1, 0, 0), scale=0.02 * scale)

    def Random_choose_N(self, N):
        """
        Choose N satellits randomly.
        Parameters
        ----------
        N - How mauny satellites to choose:int

        returns:
        N views: class MultiViewProjection(Projection)
        """

        idx = np.random.choice(self._num_projections, N, replace=False)
        # generate N unique numbers in that interval of (self._num_projections)
        new_names = [self._names[idx_i] for idx_i in idx]
        NEW = SpaceMultiView()
        NEW.set_satellites(self._sat_positions[idx], self._lookat_list[idx])
        NEW.set_commonCameraIntrinsics(self._fov_list[idx[0]], self._nx_list[idx[0]], self._ny_list[idx[0]])
        NEW.update_views(new_names)

        return NEW

    @property
    def projection_names(self):
        return self._names

    @property
    def projection(self):
        return self._names


def Create(SATS_NUMBER=10, ORBIT_ALTITUDE=500, CAM_FOV=0.1, CAM_RES=(64, 64), SAT_LOOKATS=None, VISSETUP=False):
    """
    Create the Multiview setup on orbit with one camera type.
    Parameters:
    Camera parameters are CAM_FOV, CAM_RES.
    ORBIT_ALTITUDE in km.
    SAT_LOOKATS in km is where each satellite look at. Type np.array shape of (#sats,3)
    """
    sat_positions = StringOfPearls(SATS_NUMBER=SATS_NUMBER, orbit_altitude=ORBIT_ALTITUDE)
    # --------- start the multiview setup:---------------
    # set lookat list:
    if SAT_LOOKATS is None:
        sat_lookats = np.zeros([len(sat_positions), 3])

    else:

        sat_lookats = SAT_LOOKATS

    # set camera's field of view:
    FOV = CAM_FOV  # deg
    cnx, cny = CAM_RES

    MV = SpaceMultiView()
    MV.set_satellites(sat_positions, sat_lookats)
    MV.set_commonCameraIntrinsics(FOV, cnx, cny)
    MV.update_views()

    # visualization params:
    scale = 500
    axisWidth = 0.02
    axisLenght = 5000
    if (VISSETUP):
        MV.show_setup(scale=scale, axisWidth=axisWidth, axisLenght=axisLenght, FullCone=True)
        figh = mlab.gcf()
        mlab.orientation_axes(figure=figh)
        mlab.show()

    return MV


def main():
    sat_positions = StringOfPearls()
    # --------- start the multiview setup:---------------
    # set lookat list:
    sat_lookats = np.zeros([len(sat_positions), 3])
    # set camera's field of view:
    FOV = 0.1  # deg
    cnx, cny = (64, 64)

    MV = SpaceMultiView()
    MV.set_satellites(sat_positions, sat_lookats)
    MV.set_commonCameraIntrinsics(FOV, cnx, cny)
    MV.update_views()

    # visualization params:
    scale = 500
    axisWidth = 0.02
    axisLenght = 5000
    VISSETUP = True
    if (VISSETUP):
        MV.show_setup(scale=scale, axisWidth=axisWidth, axisLenght=axisLenght, FullCone=True)
        figh = mlab.gcf()
        mlab.orientation_axes(figure=figh)

    if (VISSETUP):
        mlab.show()


if __name__ == '__main__':
    main()