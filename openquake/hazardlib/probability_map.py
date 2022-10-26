# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (c) 2016-2022 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.
import copy
import numpy
import pandas
from openquake.baselib.python3compat import zip
from openquake.baselib.performance import numba, compile

U16 = numpy.uint16
U32 = numpy.uint32
F32 = numpy.float32
F64 = numpy.float64
BYTES_PER_FLOAT = 8
poes_dt = {'gid': U16, 'sid': U32, 'lid': U16, 'poe': F64}


if numba:
    @compile("void(float64[:, :], float64[:], uint32[:])")
    def combine_probs(array, other, rlzs):
        for li in range(len(array)):
            for ri in rlzs:
                if other[li] != 0.:
                    array[li, ri] = (
                        1. - (1. - array[li, ri]) * (1. - other[li]))
else:
    def combine_probs(array, other, rlzs):
        for r in rlzs:
            array[:, r] = (1. - (1. - array[:, r]) * (1. - other))


class AllEmptyProbabilityMaps(ValueError):
    """
    Raised by get_shape(pmaps) if all passed probability maps are empty
    """


class ProbabilityCurve(object):
    """
    This class is a small wrapper over an array of PoEs associated to
    a set of intensity measure types and levels. It provides a few operators,
    including the complement operator `~`

    ~p = 1 - p

    and the inclusive or operator `|`

    p = p1 | p2 = ~(~p1 * ~p2)

    Such operators are implemented efficiently at the numpy level, by
    dispatching on the underlying array.

    Here is an example of use:

    >>> poe = ProbabilityCurve(numpy.array([0.1, 0.2, 0.3, 0, 0]))
    >>> ~(poe | poe) * .5
    <ProbabilityCurve
    [0.405 0.32  0.245 0.5   0.5  ]>
    """
    def __init__(self, array):
        self.array = array

    def __or__(self, other):
        if other == 0:
            return self
        else:
            return self.__class__(1. - (1. - self.array) * (1. - other.array))
    __ror__ = __or__

    def __iadd__(self, other):
        # this is used when composing mutually exclusive probabilities
        self.array += other.array
        return self

    def __add__(self, other):
        # this is used when composing mutually exclusive probabilities
        self.array += other.array
        return self.__class__(self.array)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.array * other.array)
        elif other == 1:
            return self
        else:
            return self.__class__(self.array * other)
    __rmul__ = __mul__

    def __pow__(self, n):
        return self.__class__(self.array ** n)

    def __invert__(self):
        return self.__class__(1. - self.array)

    def __bool__(self):
        return bool(self.array.any())

    def __repr__(self):
        return '<ProbabilityCurve\n%s>' % self.array

    def extract(self, inner_idx):
        """
        Extracts the component specified by the index `inner_idx`.
        """
        array = self.array[:, inner_idx].reshape(-1, 1)
        return self.__class__(array)

    def combine(self, other, rlz_groups):
        """
        Update a ProbabilityCurve with shape (L, R) with a pcurve with shape
        (L, G), being G the number of realization groups, which are list
        of integers in the range 0..R-1.
        """
        for g, rlz_group in enumerate(rlz_groups):
            combine_probs(self.array, other.array[:, g], rlz_group)

    # used when exporting to HDF5
    def convert(self, imtls, idx=0):
        """
        Convert a probability curve into a record of dtype `imtls.dt`.

        :param imtls: DictArray instance
        :param idx: extract the data corresponding to the given inner index
        """
        curve = numpy.zeros(1, imtls.dt)
        for imt in imtls:
            curve[imt] = self.array[imtls(imt), idx]
        return curve[0]


class Pmap(object):
    def __init__(self, sids, shape_y, shape_z):
        self.sids = sids
        self.shape_x = len(sids)
        self.shape_y = shape_y
        self.shape_z = shape_z

    def new(self, array):
        new = copy.copy(self)
        new.array = array
        return new

    def fill(self, value):
        assert 0 <= value <= 1, value
        self.array = numpy.empty((self.shape_x, self.shape_y, self.shape_z))
        self.array.fill(value)
        self.sidx = numpy.zeros(self.sids.max() + 1, numpy.uint32)
        for idx, sid in enumerate(self.sids):
            self.sidx[sid] = idx
        return self

    def reshape(self, N, M, P):
        """
        :returns: a new Pmap associated to a reshaped array
        """
        return self.new(self.array.reshape(N, M, P))

    # used in calc_hazard_curves
    def convert(self, imtls, nsites, idx=0):
        """
        Convert a probability map into a composite array of length `nsites`
        and dtype `imtls.dt`.

        :param imtls:
            DictArray instance
        :param nsites:
            the total number of sites
        :param idx:
            index on the z-axis (default 0)
        """
        curves = numpy.zeros(nsites, imtls.dt)
        for imt in curves.dtype.names:
            curves[imt][self.sids] = self.array[:, imtls(imt), idx]
        return curves

    def remove_zeros(self):
        ok = self.array.sum(axis=(1, 2)) > 0
        if ok.sum() == 0:  # avoid empty array
            ok = slice(0, 1)
        new = self.__class__(self.sids[ok], self.shape_y, self.shape_z).fill(0)
        new.array = self.array[ok]
        return new

    # used in classical_risk from CSV
    def to_dframe(self):
        """
        :returns: a DataFrame with fields sid, gid, lid, poe
        """
        dic = dict(sid=[], gid=[], lid=[], poe=[])
        for sid, arr in zip(self.sids, self.array):
            for (lid, gid), poe in numpy.ndenumerate(arr):
                dic['sid'].append(sid)
                dic['gid'].append(gid)
                dic['lid'].append(lid)
                dic['poe'].append(poe)
        for key, dt in poes_dt.items():
            dic[key] = dt(dic[key])
        dic['poe'][dic['poe'] == 1.] = .9999999999999999  # avoids log(0)
        return pandas.DataFrame(dic)

    def __ior__(self, other):
        if (other.shape_y, other.shape_z) != (self.shape_y, self.shape_z):
            raise ValueError('%s has inconsistent shape with %s' %
                             (other, self))
        if len(self.sids) == len(other.sids):
            self.array[:] = 1. - (1. - self.array) * (1. - other.array)
            return self
        # also assume other.sids are a subset of self.sids
        arr = self.array
        for sid, arr2 in zip(other.sids, other.array):
            arr[sid] = 1. - (1. - arr[sid]) * (1. - arr2)
        return self

    def __invert__(self):
        return self.new(1. - self.array)

    def __pow__(self, n):
        return self.new(self.array ** n)

    def __repr__(self):
        return '<Pmap(%d, %d, %d)>' % (self.shape_x, self.shape_y, self.shape_z)
