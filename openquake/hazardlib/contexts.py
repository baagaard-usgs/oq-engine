# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2018-2021 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import os
import abc
import copy
import time
import logging
import warnings
import itertools
import functools
import collections
import numpy
import pandas
from scipy.interpolate import interp1d
try:
    import numba
except ImportError:
    numba = None
from openquake.baselib import hdf5, parallel
from openquake.baselib.general import (
    AccumDict, DictArray, groupby, RecordBuilder, gen_slices)
from openquake.baselib.performance import Monitor
from openquake.baselib.python3compat import decode
from openquake.hazardlib import imt as imt_module
from openquake.hazardlib.const import StdDev
from openquake.hazardlib.tom import registry
from openquake.hazardlib.site import site_param_dt
from openquake.hazardlib.stats import _truncnorm_sf
from openquake.hazardlib.calc.filters import MagDepDistance
from openquake.hazardlib.probability_map import ProbabilityMap
from openquake.hazardlib.geo.surface import PlanarSurface

STD_TYPES = (StdDev.TOTAL, StdDev.INTER_EVENT, StdDev.INTRA_EVENT)
KNOWN_DISTANCES = frozenset(
    'rrup rx ry0 rjb rhypo repi rcdpp azimuth azimuth_cp rvolc closest_point'
    .split())


class Timer(object):
    """
    Timer used to save the time needed to process each source and to
    postprocess it with ``Timer('timer.csv').read_df()``. To use it, run
    the calculation on a single machine with

    OQ_TIMER=timer.csv oq run job.ini
    """
    fields = ['source_id', 'code', 'effrups', 'nsites', 'weight',
              'numctxs', 'numsites', 'dt', 'task_no']

    def __init__(self, fname):
        self.fname = fname

    def save(self, src, numctxs, numsites, dt, task_no):
        # save the source info
        if self.fname:
            row = [src.source_id, src.code.decode('ascii'),
                   src.num_ruptures, src.nsites, src.weight,
                   numctxs, numsites, dt, task_no]
            open(self.fname, 'a').write(','.join(map(str, row)) + '\n')

    def read_df(self):
        # method used to postprocess the information
        df = pandas.read_csv(self.fname, names=self.fields, index_col=0)
        df['speed'] = df['weight'] / df['dt']
        return df.sort_values('dt')


# object used to measure the time needed to process each source
timer = Timer(os.environ.get('OQ_TIMER'))


def get_distances(rupture, sites, param):
    """
    :param rupture: a rupture
    :param sites: a mesh of points or a site collection
    :param param: the kind of distance to compute (default rjb)
    :returns: an array of distances from the given sites
    """
    if not rupture.surface:  # PointRupture
        dist = rupture.hypocenter.distance_to_mesh(sites)
    elif param == 'rrup':
        dist = rupture.surface.get_min_distance(sites)
    elif param == 'rx':
        dist = rupture.surface.get_rx_distance(sites)
    elif param == 'ry0':
        dist = rupture.surface.get_ry0_distance(sites)
    elif param == 'rjb':
        dist = rupture.surface.get_joyner_boore_distance(sites)
    elif param == 'rhypo':
        dist = rupture.hypocenter.distance_to_mesh(sites)
    elif param == 'repi':
        dist = rupture.hypocenter.distance_to_mesh(sites, with_depths=False)
    elif param == 'rcdpp':
        dist = rupture.get_cdppvalue(sites)
    elif param == 'azimuth':
        dist = rupture.surface.get_azimuth(sites)
    elif param == 'azimuth_cp':
        dist = rupture.surface.get_azimuth_of_closest_point(sites)
    elif param == 'closest_point':
        t = rupture.surface.get_closest_points(sites)
        dist = numpy.vstack([t.lons, t.lats, t.depths]).T  # shape (N, 3)
    elif param == "rvolc":
        # Volcanic distance not yet supported, defaulting to zero
        dist = numpy.zeros_like(sites.lons)
    else:
        raise ValueError('Unknown distance measure %r' % param)
    dist.flags.writeable = False
    return dist


class FarAwayRupture(Exception):
    """Raised if the rupture is outside the maximum distance for all sites"""


def get_num_distances(gsims):
    """
    :returns: the number of distances required for the given GSIMs
    """
    dists = set()
    for gsim in gsims:
        dists.update(gsim.REQUIRES_DISTANCES)
    return len(dists)


def use_recarray(gsims):
    """
    :returns:
        True if the `ctx` argument of gsim.compute is a recarray for all gsims
    """
    return all(gsim.compute.__annotations__.get("ctx") is numpy.recarray
               for gsim in gsims)


def csdict(M, N, P, start, stop):
    """
    :param M: number of IMTs
    :param N: number of sites
    :param P: number of IMLs
    :param start: index
    :param stop: index > start
    """
    ddic = {}
    for _g in range(start, stop):
        ddic[_g] = AccumDict({'_c': numpy.zeros((M, N, 2, P)),
                              '_s': numpy.zeros((N, P))})
    return ddic


class ContextMaker(object):
    """
    A class to manage the creation of contexts and to compute mean/stddevs
    and possibly PoEs.

    :param trt: a tectonic region type string
    :param gsims: a list of GSIMs or a dictionary gsim -> rlz indices
    :param param:
       a dictionary of parameters like the maximum_distance, the IMTLs,
       the investigation time, etc

    NB: the trt can be different from the tectonic region type for which
    the underlying GSIMs are defined. This is intentional.
    """
    REQUIRES = ['DISTANCES', 'SITES_PARAMETERS', 'RUPTURE_PARAMETERS']
    rup_indep = True
    tom = None

    @property
    def dtype(self):
        """
        :returns: dtype of the underlying ctx_builder
        """
        return self.ctx_builder.dtype

    def __init__(self, trt, gsims, param, monitor=Monitor()):
        param = param
        self.af = param.get('af', None)
        self.max_sites_disagg = param.get('max_sites_disagg', 10)
        self.disagg_by_src = param.get('disagg_by_src')
        self.collapse_level = param.get('collapse_level', False)
        self.disagg_by_src = param.get('disagg_by_src', False)
        self.trt = trt
        self.gsims = gsims
        self.maximum_distance = (
            param.get('maximum_distance') or MagDepDistance({}))
        self.minimum_distance = param.get('minimum_distance', 0)
        self.investigation_time = param.get('investigation_time')
        if self.investigation_time:
            self.tom = registry['PoissonTOM'](self.investigation_time)
        self.ses_seed = param.get('ses_seed', 42)
        self.ses_per_logic_tree_path = param.get('ses_per_logic_tree_path', 1)
        self.trunclevel = param.get('truncation_level')
        self.num_epsilon_bins = param.get('num_epsilon_bins', 1)
        self.cross_correl = param.get('cross_correl')
        self.grp_id = param.get('grp_id', 0)
        self.effect = param.get('effect')
        self.use_recarray = use_recarray(gsims)
        for req in self.REQUIRES:
            reqset = set()
            for gsim in gsims:
                reqset.update(getattr(gsim, 'REQUIRES_' + req))
            setattr(self, 'REQUIRES_' + req, reqset)
        # self.pointsource_distance is a dict mag -> dist, possibly empty
        psd = param.get('pointsource_distance')
        if hasattr(psd, 'ddic'):
            self.pointsource_distance = psd.ddic.get(trt, {})
            if all(val == 0 for val in self.pointsource_distance.values()):
                self.pointsource_distance = 0
        else:
            self.pointsource_distance = {}
        if 'imtls' in param:
            self.imtls = param['imtls']
        elif 'hazard_imtls' in param:
            self.imtls = DictArray(param['hazard_imtls'])
        else:
            raise KeyError('Missing imtls in ContextMaker!')
        try:
            self.min_iml = param['min_iml']
        except KeyError:
            self.min_iml = [0. for imt in self.imtls]
        self.reqv = param.get('reqv')
        if self.reqv is not None:
            self.REQUIRES_DISTANCES.add('repi')
        reqs = (sorted(self.REQUIRES_RUPTURE_PARAMETERS) +
                sorted(self.REQUIRES_SITES_PARAMETERS) +
                sorted(self.REQUIRES_DISTANCES))
        dic = {}
        for req in reqs:
            if req in site_param_dt:
                dt = site_param_dt[req]
                if isinstance(dt, tuple):  # (string_, size)
                    dic[req] = b''
                else:
                    dic[req] = dt(0)
            else:
                dic[req] = 0.
        dic['occurrence_rate'] = numpy.float64(0)
        dic['sids'] = numpy.uint32(0)
        self.ctx_builder = RecordBuilder(**dic)
        self.loglevels = DictArray(self.imtls) if self.imtls else {}
        self.shift_hypo = param.get('shift_hypo')
        with warnings.catch_warnings():
            # avoid RuntimeWarning: divide by zero encountered in log
            warnings.simplefilter("ignore")
            for imt, imls in self.imtls.items():
                if imt != 'MMI':
                    self.loglevels[imt] = numpy.log(imls)
        self.init_monitoring(monitor)

    def init_monitoring(self, monitor):
        # instantiating child monitors, may be called in the workers
        self.ctx_mon = monitor('make_contexts', measuremem=False)
        self.gmf_mon = monitor('computing mean_std', measuremem=True)
        self.poe_mon = monitor('get_poes', measuremem=False)
        self.pne_mon = monitor('composing pnes', measuremem=False)
        self.task_no = getattr(monitor, 'task_no', 0)

    def read_ctxs(self, dstore, slc=None):
        """
        :param dstore: a DataStore instance
        :param slice: a slice of contexts with the same grp_id
        :returns: a list of contexts plus N lists of contexts for each site
        """
        sitecol = dstore['sitecol'].complete
        if slc is None:
            slc = dstore['rup/grp_id'][:] == self.grp_id
        params = {n: dstore['rup/' + n][slc] for n in dstore['rup']}
        ctxs = []
        for u in range(len(params['mag'])):
            ctx = RuptureContext()
            for par, arr in params.items():
                if par.endswith('_'):
                    par = par[:-1]
                setattr(ctx, par, arr[u])
            for par in sitecol.array.dtype.names:
                setattr(ctx, par, sitecol[par][ctx.sids])
            ctxs.append(ctx)
        return ctxs

    def recarray(self, ctxs):
        """
        :params ctxs: a list of contexts
        :returns: a recarray
        """
        C = sum(len(ctx) for ctx in ctxs)
        ra = self.ctx_builder.zeros(C).view(numpy.recarray)
        start = 0
        for ctx in ctxs:
            slc = slice(start, start + len(ctx))
            for par in self.ctx_builder.names:
                getattr(ra, par)[slc] = getattr(ctx, par)
            ra.sids[slc] = ctx.sids
            start = slc.stop
        return ra

    def get_ctx_params(self):
        """
        :returns: the interesting attributes of the context
        """
        params = {'occurrence_rate', 'sids_', 'src_id',
                  'probs_occur_', 'clon_', 'clat_', 'rrup_'}
        params.update(self.REQUIRES_RUPTURE_PARAMETERS)
        for dparam in self.REQUIRES_DISTANCES:
            params.add(dparam + '_')
        return params

    def from_srcs(self, srcs, sitecol):  # used in disagg.disaggregation
        """
        :param srcs: a list of Source objects
        :param sitecol: a SiteCollection instance
        :returns: a list RuptureContexts
        """
        allctxs = []
        cnt = 0
        for i, src in enumerate(srcs):
            src.id = i
            rctxs = []
            for rup in src.iter_ruptures(shift_hypo=self.shift_hypo):
                rup.rup_id = cnt
                rctxs.append(self.make_rctx(rup))
                cnt += 1
            allctxs.extend(self.get_ctxs(rctxs, sitecol, src.id))
        return allctxs

    def filter(self, sites, rup):
        """
        Filter the site collection with respect to the rupture.

        :param sites:
            Instance of :class:`openquake.hazardlib.site.SiteCollection`.
        :param rup:
            Instance of
            :class:`openquake.hazardlib.source.rupture.BaseRupture`
        :returns:
            (filtered sites, distance context)
        """
        distances = get_distances(rup, sites, 'rrup')
        mdist = self.maximum_distance(self.trt, rup.mag)
        mask = distances <= mdist
        if mask.any():
            sites, distances = sites.filter(mask), distances[mask]
        else:
            raise FarAwayRupture('%d: %d km' % (rup.rup_id, distances.min()))
        return sites, DistancesContext([('rrup', distances)])

    def make_rctx(self, rupture):
        """
        Add .REQUIRES_RUPTURE_PARAMETERS to the rupture
        """
        ctx = RuptureContext()
        vars(ctx).update(vars(rupture))
        for param in self.REQUIRES_RUPTURE_PARAMETERS:
            if param == 'mag':
                value = rupture.mag
            elif param == 'strike':
                value = rupture.surface.get_strike()
            elif param == 'dip':
                value = rupture.surface.get_dip()
            elif param == 'rake':
                value = rupture.rake
            elif param == 'ztor':
                value = rupture.surface.get_top_edge_depth()
            elif param == 'hypo_lon':
                value = rupture.hypocenter.longitude
            elif param == 'hypo_lat':
                value = rupture.hypocenter.latitude
            elif param == 'hypo_depth':
                value = rupture.hypocenter.depth
            elif param == 'width':
                value = rupture.surface.get_width()
            else:
                raise ValueError('%s requires unknown rupture parameter %r' %
                                 (type(self).__name__, param))
            setattr(ctx, param, value)
        return ctx

    def get_ctxs(self, ruptures, sitecol, src_id, mon=Monitor()):
        """
        :param ruptures:
            a list of ruptures generated by the same source
        :param sitecol:
            a (filtered) SiteCollection
        :param src_id:
            the ID of the source (for debugging purposes)
        :param mon:
            a Monitor object
        :returns:
            fat RuptureContexts
        """
        ctxs = []
        fewsites = len(sitecol.complete) <= self.max_sites_disagg
        for rup in ruptures:
            with mon:
                sites = getattr(rup, 'sites', sitecol)
                try:
                    r_sites, dctx = self.filter(sites, rup)
                except FarAwayRupture:
                    continue
                ctx = self.make_rctx(rup)
                for param in self.REQUIRES_DISTANCES - {'rrup'}:
                    distances = get_distances(rup, r_sites, param)
                    setattr(dctx, param, distances)
                reqv_obj = (self.reqv.get(self.trt) if self.reqv else None)
                if reqv_obj and isinstance(rup.surface, PlanarSurface):
                    reqv = reqv_obj.get(dctx.repi, rup.mag)
                    if 'rjb' in self.REQUIRES_DISTANCES:
                        dctx.rjb = reqv
                    if 'rrup' in self.REQUIRES_DISTANCES:
                        dctx.rrup = numpy.sqrt(
                            reqv**2 + rup.hypocenter.depth**2)
                for name in r_sites.array.dtype.names:
                    setattr(ctx, name, r_sites[name])
                ctx.src_id = src_id
                for par in self.REQUIRES_DISTANCES | {'rrup'}:
                    setattr(ctx, par, getattr(dctx, par))
                if fewsites:
                    # get closest point on the surface
                    closest = rup.surface.get_closest_points(sitecol.complete)
                    ctx.clon = closest.lons[ctx.sids]
                    ctx.clat = closest.lats[ctx.sids]
            ctxs.append(ctx)
        return ctxs

    # this is used with pointsource_distance approximation for close distances,
    # when there are many ruptures affecting few sites
    def collapse_the_ctxs(self, ctxs):
        """
        Collapse contexts with similar parameters and distances.

        :param ctxs: a list of pairs (rup, dctx)
        :returns: collapsed contexts
        """
        if len(ctxs) == 1:
            return ctxs

        if self.collapse_level >= 3:  # hack, ignore everything except mag
            rrp = ['mag']
            rnd = 0  # round distances to 1 km
        else:
            rrp = self.REQUIRES_RUPTURE_PARAMETERS
            rnd = 1  # round distances to 100 m

        def params(ctx):
            lst = []
            for par in rrp:
                lst.append(getattr(ctx, par))
            for dst in self.REQUIRES_DISTANCES:
                lst.extend(numpy.round(getattr(ctx, dst), rnd))
            return tuple(lst)

        out = []
        for values in groupby(ctxs, params).values():
            out.extend(_collapse(values))
        return out

    def max_intensity(self, sitecol1, mags, dists):
        """
        :param sitecol1: a SiteCollection instance with a single site
        :param mags: a sequence of magnitudes
        :param dists: a sequence of distances
        :returns: an array of GMVs of shape (#mags, #dists)
        """
        assert len(sitecol1) == 1, sitecol1
        nmags, ndists = len(mags), len(dists)
        gmv = numpy.zeros((nmags, ndists))
        for m, d in itertools.product(range(nmags), range(ndists)):
            mag, dist = mags[m], dists[d]
            ctx = RuptureContext()
            for par in self.REQUIRES_RUPTURE_PARAMETERS:
                setattr(ctx, par, 0)
            for dst in self.REQUIRES_DISTANCES:
                setattr(ctx, dst, numpy.array([dist]))
            for par in self.REQUIRES_SITES_PARAMETERS:
                setattr(ctx, par, getattr(sitecol1, par))
            ctx.sids = sitecol1.sids
            ctx.mag = mag
            ctx.width = .01  # 10 meters to avoid warnings in abrahamson_2014
            try:
                maxmean = self.get_mean_stds([ctx])[0].max()
                # shape NM
            except ValueError:  # magnitude outside of supported range
                continue
            else:
                gmv[m, d] = numpy.exp(maxmean)
        return gmv

    def get_pmap(self, ctxs, probmap=None):
        """
        :param ctxs: a list of contexts
        :param probmap: if not None, update it
        :returns: a new ProbabilityMap if probmap is None
        """
        tom = self.tom
        rup_indep = self.rup_indep
        if probmap is None:  # create new pmap
            pmap = ProbabilityMap(self.imtls.size, len(self.gsims))
        else:  # update passed probmap
            pmap = probmap
        for ctx, poes in self.gen_poes(ctxs):
            # pnes and poes of shape (N, L, G)
            with self.pne_mon:
                pnes = get_probability_no_exceedance(ctx, poes, tom)
                for sid, pne in zip(ctx.sids, pnes):
                    probs = pmap.setdefault(sid, self.rup_indep).array
                    if rup_indep:
                        probs *= pne
                    else:  # rup_mutex
                        probs += (1. - pne) * ctx.weight
        if probmap is None:  # return the new pmap
            return ~pmap if rup_indep else pmap

    # called by gen_poes and by the GmfComputer
    def get_mean_stds(self, ctxs):
        """
        :param ctxs: a list of contexts
        :returns: an array of shape (4, G, M, N) with mean and stddevs
        """
        if not hasattr(self, 'imts'):
            self.imts = tuple(imt_module.from_string(im) for im in self.imtls)
        N = sum(len(ctx.sids) for ctx in ctxs)
        M = len(self.imtls)
        G = len(self.gsims)
        out = numpy.zeros((4, G, M, N))
        for g, gsim in enumerate(self.gsims):
            compute = gsim.__class__.compute
            start = 0
            for ctx in ctxs:
                slc = slice(start, start + len(ctx))
                compute(gsim, ctx, self.imts, *out[:, g, :, slc])
                start = slc.stop
        return out

    # see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.845.163&rep=rep1&type=pdf
    def get_cs_contrib(self, ctxs, imti, imls):
        """
        :param ctxs:
           list of contexts defined on N sites
        :param imti:
            IMT index in the range 0..M-1
        :param imls:
            P intensity measure levels for the IMT specified by the index
        :returns:
            a dictionary g_ -> key -> array where g_ is an index,
            key is the string '_c' or '_s',  and the arrays have shape
            (M, N, 2, P) or (N, P) respectively.

        Compute the contributions to the conditional spectra, in a form
        suitable for later composition.
        """
        assert self.tom
        N = len(ctxs[0].sids)
        assert all(len(ctx) == N for ctx in ctxs[1:])
        C = len(ctxs)
        G = len(self.gsims)
        M = len(self.imtls)
        P = len(imls)
        out = csdict(M, N, P, self.start, self.start + G)
        mean_stds = self.get_mean_stds(ctxs)  # (4, G, M, N*C)
        imt_ref = self.imts[imti]
        rho = numpy.array([self.cross_correl.get_correlation(imt_ref, imt)
                           for imt in self.imts])
        m_range = range(len(self.imts))
        # probs = 1 - exp(-occurrence_rates*time_span)
        probs = self.tom.get_probability_one_or_more_occurrences(
            numpy.array([ctx.occurrence_rate for ctx in ctxs]))  # shape C
        for n in range(N):
            # NB: to understand the code below, consider the case with
            # N=3 sites and C=2 contexts; then the indices N*C are
            # 0: first site
            # 1: second site
            # 2: third site
            # 3: first site
            # 4: second site
            # 5: third site
            # i.e. idxs = [0, 3], [1, 4], [2, 5] for sites 0, 1, 2
            slc = slice(n, N * C, N)  # C indices
            for g in range(G):
                mu = mean_stds[0, g, :, slc]  # shape (M, C)
                sig = mean_stds[1, g, :, slc]  # shape (M, C)
                c = out[self.start + g]['_c']
                s = out[self.start + g]['_s']
                for p in range(P):
                    eps = (imls[p] - mu[imti]) / sig[imti]  # shape C
                    poes = _truncnorm_sf(self.trunclevel, eps)  # shape C
                    ws = -numpy.log(
                        (1. - probs) ** poes) / self.investigation_time
                    s[n, p] = ws.sum()  # weights not summing up to 1
                    for m in m_range:
                        c[m, n, 0, p] = ws @ (mu[m] + rho[m] * eps * sig[m])
                        c[m, n, 1, p] = ws @ (sig[m]**2 * (1. - rho[m]**2))
        return out

    def gen_poes(self, ctxs):
        """
        :param ctxs: a list of C context objects
        :yields: pairs (ctx, array(N, L, G))
        """
        from openquake.hazardlib.site_amplification import get_poes_site
        ctxs = [ctx.roundup(self.minimum_distance) for ctx in ctxs]
        if self.use_recarray and not numpy.isnan(
                [ctx.occurrence_rate for ctx in ctxs]).any():
            # use recarrays only for poissonian sources
            ctxs = [self.recarray(ctxs)]
        L, G = self.loglevels.size, len(self.gsims)
        for ctx in ctxs:
            with self.gmf_mon:
                # a lot of memory can be consumed here, since mean_stdt size is
                # 4 x G x M x N*C; however, you can increase concurrent_tasks
                mean_stdt = self.get_mean_stds([ctx])
            if isinstance(ctx, numpy.recarray):
                slices = [slice(0, len(ctx))]
            else:
                # splitting in chunks of at most 1000 sites to save memory
                slices = gen_slices(0, len(ctx), 1000)
            sids = ctx.sids
            s = 0
            for slc in slices:
                ctx.sids = sids[slc]
                n = slc.stop - slc.start
                with self.poe_mon:
                    poes = numpy.zeros((n, L, G))
                    for g, gsim in enumerate(self.gsims):
                        ms = mean_stdt[:2, g, :, s:s+n]
                        # builds poes of shape (n, L, G)
                        if self.af:  # kernel amplification method
                            poes[:, :, g] = get_poes_site(ms, self, ctx)
                        else:  # regular case
                            poes[:, :, g] = gsim.get_poes(ms, self, ctx)
                yield ctx, poes
                s += n


# see contexts_tests.py for examples of collapse
def combine_pmf(o1, o2):
    """
    Combine probabilities of occurrence; used to collapse nonparametric
    ruptures.

    :param o1: probability distribution of length n1
    :param o2: probability distribution of length n2
    :returns: probability distribution of length n1 + n2 - 1

    >>> combine_pmf([.99, .01], [.98, .02])
    array([9.702e-01, 2.960e-02, 2.000e-04])
    """
    n1 = len(o1)
    n2 = len(o2)
    o = numpy.zeros(n1 + n2 - 1)
    for i in range(n1):
        for j in range(n2):
            o[i + j] += o1[i] * o2[j]
    return o


def _collapse(ctxs):
    # collapse a list of contexts into a single context
    if len(ctxs) < 2:  # nothing to collapse
        return ctxs
    prups, nrups, out = [], [], []
    for ctx in ctxs:
        if numpy.isnan(ctx.occurrence_rate):  # nonparametric
            nrups.append(ctx)
        else:  # parametric
            prups.append(ctx)
    if len(prups) > 1:
        ctx = copy.copy(prups[0])
        ctx.occurrence_rate = sum(r.occurrence_rate for r in prups)
        out.append(ctx)
    else:
        out.extend(prups)
    if len(nrups) > 1:
        ctx = copy.copy(nrups[0])
        ctx.probs_occur = functools.reduce(
            combine_pmf, (n.probs_occur for n in nrups))
        out.append(ctx)
    else:
        out.extend(nrups)
    return out


def print_finite_size(rups):
    """
    Used to print the number of finite-size ruptures
    """
    c = collections.Counter()
    for rup in rups:
        if rup.surface:
            c['%.2f' % rup.mag] += 1
    print(c)
    print('total finite size ruptures = ', sum(c.values()))


class PmapMaker(object):
    """
    A class to compute the PoEs from a given source
    """
    def __init__(self, cmaker, srcfilter, group):
        vars(self).update(vars(cmaker))
        self.cmaker = cmaker
        self.srcfilter = srcfilter
        self.N = len(self.srcfilter.sitecol.complete)
        self.group = group
        self.src_mutex = getattr(group, 'src_interdep', None) == 'mutex'
        self.cmaker.rup_indep = getattr(group, 'rup_interdep', None) != 'mutex'
        self.fewsites = self.N <= cmaker.max_sites_disagg

    def count_bytes(self, ctxs):
        # # usuful for debugging memory issues
        rparams = len(self.cmaker.REQUIRES_RUPTURE_PARAMETERS)
        sparams = len(self.cmaker.REQUIRES_SITES_PARAMETERS) + 1
        dparams = len(self.cmaker.REQUIRES_DISTANCES)
        nbytes = 0
        for ctx in ctxs:
            nsites = len(ctx.sids)
            nbytes += 8 * rparams
            nbytes += 8 * sparams * nsites
            nbytes += 8 * dparams * nsites
        return nbytes

    def _ruptures(self, src, filtermag=None):
        return src.iter_ruptures(
            shift_hypo=self.shift_hypo, mag=filtermag)

    def _get_ctxs(self, rups, sites, srcid):
        ctxs = self.cmaker.get_ctxs(rups, sites, srcid, self.ctx_mon)
        if self.collapse_level > 1:
            ctxs = self.cmaker.collapse_the_ctxs(ctxs)
        out = []
        for ctx in ctxs:
            self.numsites += len(ctx.sids)
            self.numctxs += 1
            if self.fewsites:  # keep the contexts in memory
                self.rupdata.append(ctx)
            out.append(ctx)
        return out

    def _make_src_indep(self):
        # sources with the same ID
        pmap = ProbabilityMap(self.imtls.size, len(self.gsims))
        # split the sources only if there is more than 1 site
        filt = (self.srcfilter.split_less if self.N == 1
                else self.srcfilter.split)
        for src, sites in filt(self.group):
            t0 = time.time()
            if self.fewsites:
                sites = sites.complete
            self.numctxs = 0
            self.numsites = 0
            rups = self._gen_rups(src, sites)
            self.cmaker.get_pmap(self._get_ctxs(rups, sites, src.id), pmap)
            dt = time.time() - t0
            self.calc_times[src.id] += numpy.array(
                [self.numctxs, self.numsites, dt])
            timer.save(src, self.numctxs, self.numsites, dt,
                       self.cmaker.task_no)
        return ~pmap if self.cmaker.rup_indep else pmap

    def _make_src_mutex(self):
        pmap = ProbabilityMap(self.imtls.size, len(self.gsims))
        for src, sites in self.srcfilter.filter(self.group):
            t0 = time.time()
            self.numctxs = 0
            self.numsites = 0
            rups = self._ruptures(src)
            pm = ProbabilityMap(self.cmaker.imtls.size, len(self.cmaker.gsims))
            self.cmaker.get_pmap(self._get_ctxs(rups, sites, src.id), pm)
            p = pm
            if self.cmaker.rup_indep:
                p = ~p
            p *= src.mutex_weight
            pmap += p
            dt = time.time() - t0
            self.calc_times[src.id] += numpy.array(
                [self.numctxs, self.numsites, dt])
            timer.save(src, self.numctxs, self.numsites, dt,
                       self.cmaker.task_no)
        return pmap

    def dictarray(self, ctxs):
        dic = {}  # par -> array
        z = numpy.zeros(0)
        for par in self.cmaker.get_ctx_params():
            pa = par[:-1] if par.endswith('_') else par
            dic[par] = numpy.array([getattr(ctx, pa, z) for ctx in ctxs])
        return dic

    def make(self):
        self.rupdata = []
        # AccumDict of arrays with 3 elements nrups, nsites, calc_time
        self.calc_times = AccumDict(accum=numpy.zeros(3, numpy.float32))
        if self.src_mutex:
            pmap = self._make_src_mutex()
        else:
            pmap = self._make_src_indep()
        dic = {'pmap': pmap,
               'rup_data': self.dictarray(self.rupdata),
               'calc_times': self.calc_times,
               'task_no': self.task_no,
               'grp_id': self.group[0].grp_id}
        if self.disagg_by_src:
            dic['source_id'] = self.group[0].source_id
        return dic

    def _gen_rups(self, src, sites):
        # yield ruptures, each one with a .sites attribute
        def rups(rupiter, sites):
            for rup in rupiter:
                rup.sites = sites
                yield rup
        bigps = getattr(src, 'location', None) and src.count_nphc() > 1
        if bigps and self.pointsource_distance == 0:
            # finite size effects are averaged always
            yield from rups(src.avg_ruptures(), sites)
        elif bigps and self.pointsource_distance:
            # finite site effects are averaged for sites over the
            # pointsource_distance from the rupture (if any)
            cdist = sites.get_cdist(src.location)
            for ar in src.avg_ruptures():
                pdist = self.pointsource_distance['%.2f' % ar.mag]
                close = sites.filter(cdist <= pdist)
                far = sites.filter(cdist > pdist)
                if self.fewsites:
                    if close is None:  # all is far, common for small mag
                        yield from rups([ar], sites)
                    else:  # something is close
                        yield from rups(self._ruptures(src, ar.mag), sites)
                else:  # many sites
                    if close is None:  # all is far
                        yield from rups([ar], far)
                    elif far is None:  # all is close
                        yield from rups(self._ruptures(src, ar.mag), close)
                    else:  # some sites are far, some are close
                        yield from rups([ar], far)
                        yield from rups(self._ruptures(src, ar.mag), close)
        else:  # just add the ruptures
            yield from rups(self._ruptures(src), sites)


class BaseContext(metaclass=abc.ABCMeta):
    """
    Base class for context object.
    """
    def __eq__(self, other):
        """
        Return True if ``other`` has same attributes with same values.
        """
        if isinstance(other, self.__class__):
            if self._slots_ == other._slots_:
                oks = []
                for s in self._slots_:
                    a, b = getattr(self, s, None), getattr(other, s, None)
                    if a is None and b is None:
                        ok = True
                    elif a is None and b is not None:
                        ok = False
                    elif a is not None and b is None:
                        ok = False
                    elif hasattr(a, 'shape') and hasattr(b, 'shape'):
                        if a.shape == b.shape:
                            ok = numpy.allclose(a, b)
                        else:
                            ok = False
                    else:
                        ok = a == b
                    oks.append(ok)
                return numpy.all(oks)
        return False


# mock of a site collection used in the tests and in the SMTK
class SitesContext(BaseContext):
    """
    Sites calculation context for ground shaking intensity models.

    Instances of this class are passed into
    :meth:`GroundShakingIntensityModel.get_mean_and_stddevs`. They are
    intended to represent relevant features of the sites collection.
    Every GSIM class is required to declare what :attr:`sites parameters
    <GroundShakingIntensityModel.REQUIRES_SITES_PARAMETERS>` does it need.
    Only those required parameters are made available in a result context
    object.
    """
    # _slots_ is used in hazardlib check_gsim and in the SMTK
    def __init__(self, slots='vs30 vs30measured z1pt0 z2pt5'.split(),
                 sitecol=None):
        self._slots_ = slots
        if sitecol is not None:
            self.sids = sitecol.sids
            for slot in slots:
                setattr(self, slot, getattr(sitecol, slot))

    # used in the SMTK
    def __len__(self):
        return len(self.sids)


class DistancesContext(BaseContext):
    """
    Distances context for ground shaking intensity models.

    Instances of this class are passed into
    :meth:`GroundShakingIntensityModel.get_mean_and_stddevs`. They are
    intended to represent relevant distances between sites from the collection
    and the rupture. Every GSIM class is required to declare what
    :attr:`distance measures <GroundShakingIntensityModel.REQUIRES_DISTANCES>`
    does it need. Only those required values are calculated and made available
    in a result context object.
    """
    _slots_ = ('rrup', 'rx', 'rjb', 'rhypo', 'repi', 'ry0', 'rcdpp',
               'azimuth', 'hanging_wall', 'rvolc')

    def __init__(self, param_dist_pairs=()):
        for param, dist in param_dist_pairs:
            setattr(self, param, dist)

    def roundup(self, minimum_distance):
        """
        If the minimum_distance is nonzero, returns a copy of the
        DistancesContext with updated distances, i.e. the ones below
        minimum_distance are rounded up to the minimum_distance. Otherwise,
        returns the original DistancesContext unchanged.
        """
        if not minimum_distance:
            return self
        ctx = DistancesContext()
        for dist, array in vars(self).items():
            small_distances = array < minimum_distance
            if small_distances.any():
                array = numpy.array(array)  # make a copy first
                array[small_distances] = minimum_distance
                array.flags.writeable = False
            setattr(ctx, dist, array)
        return ctx


def get_dists(ctx):
    """
    Extract the distance parameters from a context.

    :returns: a dictionary dist_name -> distances
    """
    return {par: dist for par, dist in vars(ctx).items()
            if par in KNOWN_DISTANCES}


def full_context(sites, rup, dctx=None):
    """
    :returns: a full RuptureContext with all the relevant attributes
    """
    self = RuptureContext()
    for par, val in vars(rup).items():
        setattr(self, par, val)
    if hasattr(sites, 'array'):  # is a SiteCollection
        for par in sites.array.dtype.names:
            setattr(self, par, sites[par])
    else:  # sites is a SitesContext
        for par, val in vars(sites).items():
            setattr(self, par, val)
    if dctx:
        for par, val in vars(dctx).items():
            setattr(self, par, val)
    return self


def get_mean_stds(gsims, ctx, imts):
    """
    :param gsims: a single GSIM or a a list of GSIMs
    :param ctx: a RuptureContext or a recarray of size N
    :param imts: a list of M IMTs
    :returns:
        an array of shape (4, M, N) obtained by applying the
        given GSIM, ctx amd imts, or an array of shape (G, 4, M, N)
    """
    imtls = {imt.string: [0] for imt in imts}
    single = hasattr(gsims, 'compute')
    cmaker = ContextMaker('*', [gsims] if single else gsims, {'imtls': imtls})
    out = cmaker.get_mean_stds([ctx])  # (4, G, M, N)
    return out[:, 0] if single else out


# mock of a rupture used in the tests and in the SMTK
class RuptureContext(BaseContext):
    """
    Rupture calculation context for ground shaking intensity models.

    Instances of this class are passed into
    :meth:`GroundShakingIntensityModel.get_mean_and_stddevs`. They are
    intended to represent relevant features of a single rupture. Every
    GSIM class is required to declare what :attr:`rupture parameters
    <GroundShakingIntensityModel.REQUIRES_RUPTURE_PARAMETERS>` does it need.
    Only those required parameters are made available in a result context
    object.
    """
    _slots_ = (
        'mag', 'strike', 'dip', 'rake', 'ztor', 'hypo_lon', 'hypo_lat',
        'hypo_depth', 'width', 'hypo_loc')

    def __init__(self, param_pairs=()):
        for param, value in param_pairs:
            setattr(self, param, value)

    def size(self):
        """
        If the context is a multi rupture context, i.e. it contains an array
        of magnitudes and it refers to a single site, returns the size of
        the array, otherwise returns 1.
        """
        nsites = len(self.sids)
        if nsites == 1 and isinstance(self.mag, numpy.ndarray):
            return len(self.mag)
        return nsites

    # used in acme_2019
    def __len__(self):
        return len(self.sids)

    def roundup(self, minimum_distance):
        """
        If the minimum_distance is nonzero, returns a copy of the
        RuptureContext with updated distances, i.e. the ones below
        minimum_distance are rounded up to the minimum_distance. Otherwise,
        returns the original.
        """
        if not minimum_distance:
            return self
        ctx = copy.copy(self)
        for dist, array in vars(self).items():
            if dist in KNOWN_DISTANCES:
                small_distances = array < minimum_distance
                if small_distances.any():
                    array = numpy.array(array)  # make a copy first
                    array[small_distances] = minimum_distance
                    array.flags.writeable = False
                setattr(ctx, dist, array)
        return ctx


def get_probability_no_exceedance(ctx, poes, tom):
    """
    Compute and return the probability that in the time span for which the
    rupture is defined, the rupture itself never generates a ground motion
    value higher than a given level at a given site.

    Such calculation is performed starting from the conditional probability
    that an occurrence of the current rupture is producing a ground motion
    value higher than the level of interest at the site of interest.
    The actual formula used for such calculation depends on the temporal
    occurrence model the rupture is associated with.
    The calculation can be performed for multiple intensity measure levels
    and multiple sites in a vectorized fashion.

    :param ctx:
        an object with attributes .occurrence_rate and possibly .probs_occur
    :param poes:
        2D numpy array containing conditional probabilities the the a
        rupture occurrence causes a ground shaking value exceeding a
        ground motion level at a site. First dimension represent sites,
        second dimension intensity measure levels. ``poes`` can be obtained
        calling the :func:`func <openquake.hazardlib.gsim.base.get_poes>`

    :param tom:
        temporal occurrence model instance, used only if the rupture
        is parametric
    """
    rate = ctx.occurrence_rate
    try:
        n = len(rate)
    except TypeError:  # float' has no len()
        if numpy.isnan(rate):  # nonparametric rupture
            # Uses the formula
            #
            #    ∑ p(k|T) * p(X<x|rup)^k
            #
            # where `p(k|T)` is the probability that the rupture occurs k times
            # in the time span `T`, `p(X<x|rup)` is the probability that a
            # rupture occurrence does not cause a ground motion exceedance, and
            # thesummation `∑` is done over the number of occurrences `k`.
            #
            # `p(k|T)` is given by the attribute probs_occur and
            # `p(X<x|rup)` is computed as ``1 - poes``.
            prob_no_exceed = numpy.float64(
                [v * (1 - poes) ** i for i, v in enumerate(ctx.probs_occur)]
            ).sum(axis=0)
            return numpy.clip(prob_no_exceed, 0., 1.)  # avoid numeric issues
        else:
            return tom.get_probability_no_exceedance(rate, poes)

    # passed a recarray context, poes has shape (n, L, G)
    assert len(poes) == n
    res = numpy.zeros_like(poes)
    for i in range(n):
        res[i] = tom.get_probability_no_exceedance(rate[i], poes[i])
    return res


class Effect(object):
    """
    Compute the effect of a rupture of a given magnitude and distance.

    :param effect_by_mag: a dictionary magstring -> intensities
    :param dists: array of distances, one per each intensity
    :param cdist: collapse distance
    """
    def __init__(self, effect_by_mag, dists, collapse_dist=None):
        self.effect_by_mag = effect_by_mag
        self.dists = dists
        self.nbins = len(dists)

    def collapse_value(self, collapse_dist):
        """
        :returns: intensity at collapse distance
        """
        # get the maximum magnitude with a cutoff at 7
        for mag in self.effect_by_mag:
            if mag > '7.00':
                break
        effect = self.effect_by_mag[mag]
        idx = numpy.searchsorted(self.dists, collapse_dist)
        return effect[idx-1 if idx == self.nbins else idx]

    def __call__(self, mag, dist):
        di = numpy.searchsorted(self.dists, dist)
        if di == self.nbins:
            di = self.nbins
        eff = self.effect_by_mag['%.2f' % mag][di]
        return eff

    # this is used to compute the magnitude-dependent pointsource_distance
    def dist_by_mag(self, intensity):
        """
        :returns: a dict magstring -> distance
        """
        dst = {}  # magnitude -> distance
        for mag, intensities in self.effect_by_mag.items():
            if intensity < intensities.min():
                dst[mag] = self.dists[-1]  # largest distance
            elif intensity > intensities.max():
                dst[mag] = self.dists[0]  # smallest distance
            else:
                dst[mag] = interp1d(intensities, self.dists)(intensity)
        return dst


def get_effect_by_mag(mags, sitecol1, gsims_by_trt, maximum_distance, imtls):
    """
    :param mags: an ordered list of magnitude strings with format %.2f
    :param sitecol1: a SiteCollection with a single site
    :param gsims_by_trt: a dictionary trt -> gsims
    :param maximum_distance: an MagDepDistance object
    :param imtls: a DictArray with intensity measure types and levels
    :returns: a dict magnitude-string -> array(#dists, #trts)
    """
    trts = list(gsims_by_trt)
    ndists = 51
    gmv = numpy.zeros((len(mags), ndists, len(trts)))
    param = dict(maximum_distance=maximum_distance, imtls=imtls)
    for t, trt in enumerate(trts):
        dist_bins = maximum_distance.get_dist_bins(trt, ndists)
        cmaker = ContextMaker(trt, gsims_by_trt[trt], param)
        gmv[:, :, t] = cmaker.max_intensity(
            sitecol1, [float(mag) for mag in mags], dist_bins)
    return dict(zip(mags, gmv))


# not used at the moment
def get_effect(mags, sitecol1, gsims_by_trt, oq):
    """
    :params mags:
       a dictionary trt -> magnitudes
    :param sitecol1:
       a SiteCollection with a single site
    :param gsims_by_trt:
       a dictionary trt -> gsims
    :param oq:
       an object with attributes imtls, minimum_intensity,
       maximum_distance and pointsource_distance
    :returns:
       an ArrayWrapper trt -> effect_by_mag_dst and a nested dictionary
       trt -> mag -> dist with the effective pointsource_distance

    Updates oq.maximum_distance.magdist
    """
    assert list(mags) == list(gsims_by_trt), 'Missing TRTs!'
    dist_bins = {trt: oq.maximum_distance.get_dist_bins(trt)
                 for trt in gsims_by_trt}
    aw = hdf5.ArrayWrapper((), {})
    # computing the effect make sense only if all IMTs have the same
    # unity of measure; for simplicity we will consider only PGA and SA
    psd = oq.pointsource_distance
    if psd is not None:
        psd.interp(mags)
        psd = psd.ddic
    if psd:
        logging.info('Computing effect of the ruptures')
        allmags = set()
        for trt in mags:
            allmags.update(mags[trt])
        eff_by_mag = parallel.Starmap.apply(
            get_effect_by_mag, (sorted(allmags), sitecol1, gsims_by_trt,
                                oq.maximum_distance, oq.imtls)
        ).reduce()
        effect = {}
        for t, trt in enumerate(mags):
            arr = numpy.array([eff_by_mag[mag][:, t] for mag in mags[trt]])
            setattr(aw, trt, arr)  # shape (#mags, #dists)
            setattr(aw, trt + '_dist_bins', dist_bins[trt])
            effect[trt] = Effect(dict(zip(mags[trt], arr)), dist_bins[trt])
        minint = oq.minimum_intensity.get('default', 0)
        for trt, eff in effect.items():
            if minint:
                oq.maximum_distance.ddic[trt] = eff.dist_by_mag(minint)
            # build a dict trt -> mag -> dst
            if psd and set(psd[trt].values()) == {-1}:
                maxdist = oq.maximum_distance(trt)
                psd[trt] = eff.dist_by_mag(eff.collapse_value(maxdist))
    return aw


def read_cmakers(dstore, full_lt=None):
    """
    :param dstore: a DataStore-like object
    :param full_lt: a FullLogicTree instance, if given
    :returns: a list of ContextMaker instance, one per source group
    """
    from openquake.hazardlib.site_amplification import AmplFunction
    cmakers = []
    oq = dstore['oqparam']
    full_lt = full_lt or dstore['full_lt']
    trt_smrs = dstore['trt_smrs'][:]
    toms = dstore['toms'][:]
    rlzs_by_gsim_list = full_lt.get_rlzs_by_gsim_list(trt_smrs)
    trts = list(full_lt.gsim_lt.values)
    num_eff_rlzs = len(full_lt.sm_rlzs)
    start = 0
    # some ugly magic on the pointsource_distance
    if oq.pointsource_distance:
        mags = dstore['source_mags']
        psd = MagDepDistance.new(str(oq.pointsource_distance))
        psd.interp({trt: mags[trt][:] for trt in mags})
        oq.pointsource_distance = psd
    for grp_id, rlzs_by_gsim in enumerate(rlzs_by_gsim_list):
        trti = trt_smrs[grp_id][0] // num_eff_rlzs
        trt = trts[trti]
        if ('amplification' in oq.inputs and
                oq.amplification_method == 'kernel'):
            df = AmplFunction.read_df(oq.inputs['amplification'])
            af = AmplFunction.from_dframe(df)
        else:
            af = None
        cmaker = ContextMaker(
            trt, rlzs_by_gsim,
            {'truncation_level': oq.truncation_level,
             'collapse_level': int(oq.collapse_level),
             'num_epsilon_bins': oq.num_epsilon_bins,
             'investigation_time': oq.investigation_time,
             'maximum_distance': oq.maximum_distance,
             'pointsource_distance': oq.pointsource_distance,
             'minimum_distance': oq.minimum_distance,
             'ses_seed': oq.ses_seed,
             'ses_per_logic_tree_path': oq.ses_per_logic_tree_path,
             'max_sites_disagg': oq.max_sites_disagg,
             'disagg_by_src': oq.disagg_by_src,
             'min_iml': oq.min_iml,
             'imtls': oq.imtls,
             'reqv': oq.get_reqv(),
             'shift_hypo': oq.shift_hypo,
             'cross_correl': oq.cross_correl,
             'af': af,
             'grp_id': grp_id})
        cmaker.tom = registry[decode(toms[grp_id])](oq.investigation_time)
        cmaker.trti = trti
        cmaker.start = start
        start += len(rlzs_by_gsim)
        cmakers.append(cmaker)
    return cmakers


def read_cmaker(dstore, trt_smr):
    """
    :param dstore: a DataStore-like object
    :returns: a ContextMaker instance
    """
    oq = dstore['oqparam']
    full_lt = dstore['full_lt']
    trts = list(full_lt.gsim_lt.values)
    trt = trts[trt_smr // len(full_lt.sm_rlzs)]
    rlzs_by_gsim = full_lt.get_rlzs_by_gsim()[trt_smr]
    mags = dstore['source_mags']
    md = MagDepDistance.new(str(oq.maximum_distance))
    md.interp({trt: mags[trt][:] for trt in mags})
    cmaker = ContextMaker(
        trt, rlzs_by_gsim,
        {'truncation_level': oq.truncation_level,
         'collapse_level': int(oq.collapse_level),
         'num_epsilon_bins': oq.num_epsilon_bins,
         'investigation_time': oq.investigation_time,
         'maximum_distance': md,
         'minimum_distance': oq.minimum_distance,
         'ses_seed': oq.ses_seed,
         'ses_per_logic_tree_path': oq.ses_per_logic_tree_path,
         'max_sites_disagg': oq.max_sites_disagg,
         'disagg_by_src': oq.disagg_by_src,
         'min_iml': oq.min_iml,
         'imtls': oq.imtls,
         'reqv': oq.get_reqv(),
         'shift_hypo': oq.shift_hypo})
    return cmaker
