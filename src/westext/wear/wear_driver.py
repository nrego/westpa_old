# Copyright (C) 2013 Matthew C. Zwier, Joshua L. Adelman and Lillian T. Chong
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

from __future__ import division; __metaclass__ = type

import logging
log = logging.getLogger(__name__)

import numpy
import operator
from itertools import izip, imap

import westpa, west
from westpa.extloader import get_object
from westpa.yamlcfg import check_bool
from west.data_manager import weight_dtype

EPS = numpy.finfo(numpy.float64).eps


class WEARDriver:
    def __init__(self, sim_manager, plugin_config):
        if not sim_manager.work_manager.is_master:
            return

        self.sim_manager = sim_manager
        self.data_manager = sim_manager.data_manager
        self.system = sim_manager.system
        self.work_manager = sim_manager.work_manager

        self.do_reweight = check_bool(plugin_config.get('do_reweighting', False))

        self.reweight_period = plugin_config.get('reweight_period', 0)
        self.priority = plugin_config.get('priority', 0)

        # Python file defining bin mapper. Use driver bin mapper if not provided
        self.binweight_file = plugin_config.get('binprob_file', None)

        if self.binweight_file is None:
            log.error('No bin weight file defined')
            raise

        try:
            self.binprobs = numpy.loadtxt(self.binweight_file, dtype=weight_dtype)
        except:
            log.error('Error loading binweight file {}'.format(self.binweight_file))
            raise

        self.binprobs /= self.binprobs.sum()

        if self.do_reweight:
            sim_manager.register_callback(sim_manager.prepare_new_iteration,self.prepare_new_iteration, self.priority)

        bin_mapper_file = plugin_config.get('bin_mapper_file', None)
        self.bin_mapper = None

        if bin_mapper_file is not None:
            try:
                self.bin_mapper = get_object(bin_mapper_file)
            except:
                log.error('ERROR loading bin mapper from file: {}'.format(bin_mapper_file))

    def prepare_new_iteration(self):
        n_iter = self.sim_manager.n_iter
        we_driver = self.sim_manager.we_driver

        if not self.do_reweight:
            # Reweighting not requested
            log.debug('arbitrary reweighting not enabled') 
            return

        with self.data_manager.lock:
            wear_global_group = self.data_manager.we_h5file.require_group('wear')
            last_reweighting = long(wear_global_group.attrs.get('last_reweighting', 0))

        if n_iter - last_reweighting < self.reweight_period:
            # Not time to reweight yet
            log.debug('not reweighting')
            return
        else:
            log.debug('reweighting')

        mapper = self.bin_mapper if self.bin_mapper is not None else we_driver.bin_mapper

        bins = we_driver.next_iter_binning
        n_bins = len(bins)


        with self.data_manager.flushing_lock():
            orig_binprobs = numpy.fromiter(imap(operator.attrgetter('weight'),bins), dtype=numpy.float64, count=n_bins)

        westpa.rc.pstatus('\nBin probabilities prior to reweighting:\n{!s}'.format(orig_binprobs))
        westpa.rc.pflush()

        binprobs = self.binprobs

        if self.bin_mapper is not None:
            segments = list(we_driver.next_iter_segments)
            pcoords = numpy.empty((len(segments), self.system.pcoord_ndim), dtype=self.system.pcoord_dtype)
            
            for iseg, segment in enumerate(segments):
                pcoords[iseg] = segment.pcoord[0,:]

            assignments = mapper.assign(pcoords)

            bins = mapper.construct_bins()
            n_bins = len(bins)

            for (segment, idx) in izip(segments, assignments):
                bins[idx].add(segment)

            orig_binprobs = numpy.fromiter(imap(operator.attrgetter('weight'),bins), dtype=numpy.float64, count=n_bins)

        # Check to see if reweighting has set non-zero bins to zero probability (should never happen)
        assert (~((orig_binprobs > 0) & (binprobs == 0))).all(), 'populated bin reweighted to zero probability'
        
        # Check to see if reweighting has set zero bins to nonzero probability (may happen)
        z2nz_mask = (orig_binprobs == 0) & (binprobs > 0) 
        if (z2nz_mask).any():
            westpa.rc.pstatus('Reweighting would assign nonzero probability to an empty bin; not reweighting this iteration.')
            westpa.rc.pstatus('Empty bins assigned nonzero probability: {!s}.'
                                .format(numpy.array_str(numpy.arange(n_bins)[z2nz_mask])))
        else:
            for (bin, newprob) in izip(bins, binprobs):
                bin.reweight(newprob)

            final_bins = we_driver.next_iter_binning
            n_bins_final = len(final_bins)
            final_binprobs = numpy.fromiter(imap(operator.attrgetter('weight'),final_bins), dtype=numpy.float64, count=n_bins_final)
            westpa.rc.pstatus('\nBin populations after reweighting:\n{!s}'.format(final_binprobs))
            wear_global_group.attrs['last_reweighting'] = n_iter

        assert (abs(1 - numpy.fromiter(imap(operator.attrgetter('weight'),bins), dtype=numpy.float64, count=n_bins).sum())
                < EPS * numpy.fromiter(imap(len,bins), dtype=numpy.int, count=n_bins).sum())

        westpa.rc.pflush()
