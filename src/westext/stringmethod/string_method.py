# Copyright (C) 2013 Joshua L. Adelman
#
# This file is part of WESTPA.
#
# WESTPA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# WESTPA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with WESTPA.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from fourier_fitting import FourierFit
from collections import Iterable

try:
    import scipy
    import scipy.interpolate
    import scipy.linalg
    SCIPY_FLAG = True
except:
    SCIPY_FLAG = False

from westext.stringmethod import WESTStringMethod
from westpa.extloader import get_object

import logging
log = logging.getLogger(__name__)


class DefaultStringMethod(WESTStringMethod):
    """ Implementation of a method to evolve one or more pseudo-1D strings in a high dimensional
        progress coordinate space.

        **Parameters**
        centers:        A numpy array of size (number of total centers,pcoord dim) that stores
                            the positions of all of the string images
        slen:           An iterable containing the number of centers in each string
        slabels:        An list containing the relative positions in each string of any state label
                            progress coordinates if present. These progress coordinates will be ignored in the
                            calculation. None if no labels
        mpairs:         A list of lists containing the indices of pairs of centers that should move together.
                            None if strings move independently 
        dtau:           Parameter controlling the rate at which centers move toward the average value in the bin
        kappa:          Parameter controlling the smoothing of the string
        fixed_ends:     Boolean flag specifying whether to fix ends of the strings
        sciflag:        Boolean flag specifying whether to attempt to use scipy methods which are
                            generally more efficient
        fourierflag:    Boolean flag specifying whether to user fourier fitting method
        fourier_P:      Integer value specifying how many fourier modes to use in fitting
        fourier_maxiters: Maximum number of iterations of fourier fitting procedure
        fourier_tol:    Tolerance for ending fourier fitting
    """

    def __init__(self, centers, slen=None, slabels=None, mpairs=None, dtau=0.1,
                    kappa=0.1, sciflag=None, fixed_ends=True, 
                    fourierflag=False, fourier_P=2, fourier_maxiters=100, fourier_tol=1.0E-6, **kwargs):
        super(DefaultStringMethod, self).__init__(centers, **kwargs)

        self._SCIPY_FLAG = None
        self._fixed_ends = fixed_ends

        if sciflag is None:
            self._SCIPY_FLAG = SCIPY_FLAG
        else:
            self._SCIPY_FLAG = SCIPY_FLAG & sciflag

        
        self._ENERGY_WEIGHT_FLAG = kwargs.get('energy_weight', False)

        self._TRUNCATE_FLAG = kwargs.get('truncate', False)

        if self._ENERGY_WEIGHT_FLAG and self._TRUNCATE_FLAG:
            log.error('ERROR: energy weighting and truncation cannot both be on')
            raise

        self._energy_cutoff = kwargs.get('truncation_energy', None)
        # Custom function that gives energy (or any value with which to cutoff)
        #   as a function of progress coordinate space
        try:
            self._cutoff_function = get_object(kwargs.get('truncation_func', None))
        except:
            log.info('No cutoff function supplied')
            self._cutoff_function = None

        if self._energy_cutoff is None and self._cutoff_function is not None:
            log.error('ERROR: need energy cutoff and cutoff function for truncation based reweighting')
            raise

        self._dtau = dtau
        self._kappa = kappa

        self._centers = centers

        self._nstrings = len(slen)      # Number of strings
        self._N = centers.shape[0]
        self._ndim = centers.shape[1]    # Number of progress coordinates

        if slen and isinstance(slen, Iterable):
            self._slen = np.array(slen)
        elif slen:
            self._slen = np.array([slen])
        else:
            self._slen = np.array([self._N])
            log.warning('Input parameters do not define slen; assuming system is composed of a single string')

        self._mpairs = mpairs

        assert np.sum(self._slen) == self._N

        self._skip_dim = np.array(slabels) if slabels is not None else np.array([])

        # Get iterable of slicing objects to get per string center coordinates to perform calculation on
        self._strindx = []

        indx_all = np.arange(self._ndim)
        self._indx_take = np.setdiff1d(indx_all,self._skip_dim)
        self._ndim_take = len(self._indx_take)

        start_count = 0
        for sl in slen:
            self._strindx.append(np.index_exp[start_count:(start_count+sl),self._indx_take])
            start_count += sl

        # Fourier fitting parameters
        self._FFIT_FLAG = fourierflag
        self._ffp = fourier_P
        self._ffmaxiters = fourier_maxiters
        self._fftol = fourier_tol

        # Create dict to hold kappan and A objects for all unique lengths of strings
        self._kappan = {}
        self._A = {}

        self.finalize_init()

    @property
    def centers(self):
        return self._centers

    @property
    def length(self):
        L = []
        for sid,si in enumerate(self._strindx):
            L.append(self.calculate_length(self._centers[si])[-1])

        return L

    def calculate_length(self,x):
        dd = x - np.roll(x, 1, axis=0)
        dd[0,:] = 0.0
        return np.cumsum(np.sqrt((dd*dd).sum(axis=1)))

    def finalize_init(self):
        # Set up A and kappan for each string
        uslen = np.unique(self._slen)

        for ulen in uslen:
            self._kappan[ulen] = self._kappa * self._dtau * ulen
            self._A[ulen] = None

            if self._SCIPY_FLAG:
                ud = np.zeros((ulen,))
                ld = np.zeros((ulen,))
                d = np.ones((ulen,))

                d[1:-1] = 2.0*self._kappan[ulen] + 1.0
                ud[2:] = -self._kappan[ulen]
                ld[:-2] = -self._kappan[ulen]

                self._A[ulen] = np.mat([ud,d,ld])

            else:
                self._A[ulen] = np.eye(ulen)
                di = np.diag_indices(ulen, ndim=2)
                ii = (di[0][1:-1],di[1][1:-1])

                self._A[ulen][ii] = 2.0*self._kappan[ulen] + 1.0

                dd = np.zeros((ulen-1,))
                dd[1:] = -self._kappan[ulen]
                self._A[ulen] += np.diag(dd,k=1)

                dd = np.zeros((ulen-1,))
                dd[:-1] = -self._kappan[ulen]
                self._A[ulen] += np.diag(dd,k=-1)

    def update_string_centers(self, avgcoords, binprob):
        """ Update the position of all string centers
        **Parameters**
        avgcoords:      Average position of replicas in each voronoi cell
        binprob:        The total weight in each voronoi cell

        """
        assert self.centers.shape == avgcoords.shape

        # If centers are paired, calculate their average position
        if self._mpairs is not None:
            for pi in self._mpairs:
                pprob = binprob[pi]
                if np.sum(pprob) == 0.0:
                    continue
                
                idx = np.ix_(pi,self._indx_take)
                pavg = np.ma.array(avgcoords[idx])
                zind = np.where(pprob == 0)[0]
                pavg[zind,:] = np.ma.masked

                avgcoords[idx] = pavg.mean(axis=0)

        for sid,si in enumerate(self._strindx):

            x = avgcoords.copy()[si]
            centers = self.centers[si]
            occupied = np.nonzero(binprob[si[0]])

            N = self._slen[sid]

            # if avgcoords has missing values fill them by linearly interpolating
            # present data
            if occupied[0].shape != N:
                notocc = np.ones((N,),dtype=np.bool)  # unoccupied
                notocc[occupied] = False
                
                # marked paired centers as occupied to avoid reseting averaged value if 
                # at least one is occupied
                if self._mpairs is not None:
                    for pi in self._mpairs:
                        totprob = np.sum(binprob[pi])
                        if totprob == 0.0:
                            continue
                        else:
                            for m in pi:
                                if (m >= si[0].start) and (m < si[0].stop):
                                    notocc[m-si[0].start] = False 
                
                cfunc = lambda z: z.nonzero()[0]

                # Handle ends first
                if notocc[0]:
                    x[0,:] = centers[0,:]
                    notocc[0] = False
                if notocc[-1]:
                    x[-1,:] = centers[-1,:] 
                    notocc[-1] = False

                # interpolate values for unoccupied bins
                if self._SCIPY_FLAG:
                    for k in xrange(self._ndim_take):
                        f = scipy.interpolate.interp1d(cfunc(~notocc),x[~notocc,k],kind='linear')
                        x[notocc,k] = f(cfunc(notocc))
                else:
                    for k in xrange(self._ndim_take):
                        x[notocc,k] = np.interp(cfunc(notocc),cfunc(~notocc),x[~notocc,k])

            if self._fixed_ends:
                x[0,:] = centers[0,:]
                x[-1,:] = centers[-1,:]

            psi = centers
            psi_new = np.zeros_like(psi)

            b = psi - self._dtau*(psi - x)

            # Update and smooth the string
            if self._SCIPY_FLAG:
                for k in xrange(self._ndim_take):
                    psi_new[:,k] = scipy.linalg.solve_banded((1,1),self._A[N],b[:,k])
            else:
                for k in xrange(self._ndim_take):
                    psi_new[:,k] = np.linalg.solve(self._A[N],b[:,k])


            # Optionally smooth using fourier method
            if self._FFIT_FLAG:
                w0 = np.zeros((self._ndim_take,self._ffp),np.float)
                t0 = np.linspace(0,1,psi_new.shape[0])

                ff = FourierFit(P=self._ffp,maxiters=self._ffmaxiters)
                ff.optimize(psi_new,None,w0,t0)
                psi_new = ff.pp[-1][:]

            # Enforce equal or energy-weighted spacing between centers along the string
            L = self.calculate_length(psi_new)

            endpoint_left = 0.0
            endpoint_right = L[-1]

            # Truncation based energy weighting
            if self._TRUNCATE_FLAG:

                # Hack to fill in bin probabilities for unoccupied bins
                #   with minimum occupied bin probability
                zero_bin = binprob == 0
                if zero_bin.any():
                    binprob[zero_bin] = binprob[~zero_bin].min()
                binenergy = -np.log(binprob)
                binenergy -= binenergy.min()

                #binenergy = self._cutoff_function(self.centers)
                binenergy = self._cutoff_function(psi_new)

                energy_cutoff = self._energy_cutoff

                binmask = binenergy >= energy_cutoff
                log.info('bin energies: {}'.format(binenergy))
                if binmask.any():
                    
                    # How to account for possibility of more than two crosses?
                    cross_idx = np.bitwise_xor(binmask[:-1], binmask[1:])
                    if cross_idx.sum() <= 2:
                        log.info('Attempting to truncate string')
                        crosses, = np.where(cross_idx)
                        assert crosses.size <= 2
                        if crosses.size == 2:
                            l_idx, r_idx = crosses
                            assert r_idx > l_idx

                            slope_left = (binenergy[l_idx+1]-binenergy[l_idx]) / (L[l_idx+1] - L[l_idx])
                            slope_right = (binenergy[r_idx]-binenergy[r_idx+1]) / (L[r_idx] - L[r_idx+1])

                            endpoint_left = L[l_idx] + (energy_cutoff - binenergy[l_idx])/slope_left
                            endpoint_right = L[r_idx+1] + (energy_cutoff - binenergy[r_idx+1])/slope_right
                        elif crosses.size == 1:
                            idx = crosses[0]

                            # Right cross (greater to lower energy)
                            if binmask[idx]:
                                r_idx = idx
                                slope_right = (binenergy[r_idx]-binenergy[r_idx+1]) / (L[r_idx] - L[r_idx+1])
                                endpoint_right = L[r_idx+1] + (energy_cutoff - binenergy[r_idx+1])/slope_right

                            # Left cross (lower to greater energy)
                            else:
                                l_idx = idx
                                slope_left = (binenergy[l_idx+1]-binenergy[l_idx]) / (L[l_idx+1] - L[l_idx])
                                endpoint_left = L[l_idx] + (energy_cutoff - binenergy[l_idx])/slope_left
                    else:
                        log.info('Too many minimum energy crosses ({}), not truncating'.format(cross_idx.sum()))

            if self._ENERGY_WEIGHT_FLAG:

                norm_binprob = binprob[si[0]] / binprob[si[0]].sum()

                # Hack to fill in bin probabilities for unoccupied bins
                #   with minimum occupied bin probability
                zero_bin = norm_binprob == 0
                if zero_bin.any():
                    norm_binprob[zero_bin] = norm_binprob[~zero_bin].min()

                binenergy = -np.log(norm_binprob)
                binenergy -= binenergy.min()
                # Average energy in each bin from 2 to N
                avgenergy = (binenergy[:-1]+binenergy[1:]) / 2.0

                #avgenergy = avgenergy**2

                gradenergy = np.abs(np.diff(binenergy))

                L[1:] = np.cumsum(avgenergy*np.diff(L))

            endpoint_left /= L[-1]
            endpoint_right /= L[-1]
            L /= L[-1]
            g2 = np.linspace(endpoint_left,endpoint_right,N)

            if self._SCIPY_FLAG:
                for k in xrange(self._ndim_take):
                    f = scipy.interpolate.interp1d(L,psi_new[:,k],kind='linear')
                    psi_new[:,k] = f(g2)
            else:
                for k in xrange(self._ndim_take):
                    psi_new[:,k] = np.interp(g2,L,psi_new[:,k])

            self.centers[si] = psi_new.copy()
