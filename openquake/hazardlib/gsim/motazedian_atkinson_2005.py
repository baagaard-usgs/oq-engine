# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2023, GEM Foundation
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

import numpy as np

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib.gsim.atkinson_boore_2006 import (
    AtkinsonBoore2006, _get_pga_on_rock, _get_site_amplification_linear,
    _get_site_amplification_non_linear)
from openquake.hazardlib.gsim.boore_atkinson_2008 import BooreAtkinson2008
from openquake.hazardlib import const, contexts
from openquake.hazardlib.imt import PGA, PGV, SA

#: Equation constants that are IMT-independent
CONSTS = {
    "c5": -7.333,
    "c6": 2.333,
    "c7": -1.8,
    "c8": 0.1,
    "dist_hinge_lower": 75.0,
    "dist_hinge_upper": 100.0,
}

BASE_10_TO_E = np.log(10)
LN_CM_TO_G = np.log(980.0)


def _compute_r(rrup, mag):
    delta = CONSTS["c5"] + CONSTS["c6"]*mag
    return np.sqrt(rrup**2 + delta**2)


def _compute_magnitude_scaling(mag, C):
    dmag = mag - 6.0
    return C["c1"] + C["c2"] * dmag + C["c3"]*dmag**2


def _compute_hinge_function(mag, r):
    hinge_factor = CONSTS["c7"] + CONSTS["c8"]*mag

    res = np.zeros_like(r)

    idx = r <= CONSTS["dist_hinge_lower"]
    res[idx] = hinge_factor[idx] * np.log10(r[idx])

    idx = np.logical_and(r > CONSTS["dist_hinge_lower"], r <= CONSTS["dist_hinge_upper"])
    res[idx] = hinge_factor[idx] * np.log10(CONSTS["dist_hinge_lower"])

    idx = r > 100.0
    res[idx] = hinge_factor[idx] * np.log10(CONSTS["dist_hinge_lower"]) - 0.5*np.log10(r[idx]/CONSTS["dist_hinge_upper"])

    return res


def _compute_anelastic_attenuation(C, r):
    return C["c4"]*r


class MotazedianAtkinson2005(GMPE):
    """
    Implements GMPE developed by Dariush Motazedian and Gail M. Atkinson, and
    published as "Ground-motion relations for Puerto Rico", in P. Mann (Ed.),
    Active Tectonics and Seismic Hazards of Puerto Rico, the Virgin Islands,
    and Offshore Areas (Vol. 385). Geological Society of America.
    DOI: 10.1130/0-8137-2385-X.61.

    log_10 Y (f, R) = c1 + c2 * (M-6) + c3*(M-6)**2 + hingeFunction + c4(f)**R

    R = (D**2 + Δ**2)**0.5
    D = closest distance to fault surface, in km
    Δ = -7.333 + 2.333*M
    hingeFunction = (-1.8 + 0.1*M) log (R) for R ≤ 75 km
    hingeFunction = (-1.8 + 0.1*M) log (75) for 75 km ≤R ≤ 100 km
    hingeFunction = (-1.8 + 0.1*M) log (75) - 0.5log (R/100) for R ≥ 100 km
    c4(f) is the coefficient of anelastic attenuation.
    """
    #: Supported tectonic region type is active shallow crust, see
    #: Table 1, page 23.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration,
    #: peak ground velocity and peak ground acceleration, see Table 2
    #: pg. 30.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    #: Supported intensity measure component is orientation-independent
    #: measure :attr:`~openquake.hazardlib.const.IMC.GMRotI50.
    #: Manuscript does not specify the component, so we assume it is RotD50.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GMRotI50

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see second paragraph on page 14.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}

    #: Required rupture parameter is magnitude.
    #: See equation 6 on page 12.
    REQUIRES_RUPTURE_PARAMETERS = {'mag'}

    #: Assumes generic soft rock (NEHRP B or C).
    REQUIRES_SITES_PARAMETERS = {}

    #: Required distance measure is Rrup.
    #: See 'Ground-motion relations' page 12.
    REQUIRES_DISTANCES = {'rrup'}

    #: Shear-wave velocity for reference soil conditions in [m s-1]
    DEFINED_FOR_REFERENCE_VELOCITY = 760.

    kind = 'base'

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]

            # Equation 7, page 13.
            R = _compute_r(ctx.rrup, ctx.mag)
            mean[m] = _compute_magnitude_scaling(ctx.mag, C) + \
                _compute_hinge_function(ctx.mag, R) + \
                _compute_anelastic_attenuation(C, R)
            mean[m] *= BASE_10_TO_E
            mean[m] -= LN_CM_TO_G

            sig[m] = 0.28 * BASE_10_TO_E

# Use PGA for SA(T=0.01s)
    COEFFS = CoeffsTable(sa_damping=5, table="""\
IMT       f    c1       c2        c3        c4
PGA    0.00  3.60  0.35181  -0.06926  -0.00201
PGV   -1.00  2.35  0.54828  -0.06350  -0.00107
0.01   0.01  3.60  0.35181  -0.06926  -0.00201
0.06  15.85  3.88  0.33249  -0.06818  -0.00251
0.08  12.59  3.94  0.32165  -0.06523  -0.00253
0.1   10.00  3.96  0.32088  -0.06542  -0.00244
0.12   7.94  3.98  0.32515  -0.07216  -0.00234
0.16   6.31  3.97  0.33046  -0.07344  -0.00219
0.2    5.01  3.94  0.33077  -0.06816  -0.00204
0.25   3.98  3.88  0.35932  -0.07932  -0.00185
0.3    3.16  3.83  0.38087  -0.09045  -0.00159
0.4    2.51  3.74  0.40472  -0.08864  -0.00139
0.5    2.00  3.68  0.44246  -0.10831  -0.00126
0.6    1.59  3.58  0.47303  -0.11486  -0.00118
0.8    1.26  3.47  0.49700  -0.11945  -0.00105
1      1.00  3.35  0.56986  -0.14377  -0.00086
1.25   0.79  3.20  0.63441  -0.15706  -0.00080
1.5    0.63  3.04  0.67664  -0.15973  -0.00061
2      0.50  2.89  0.73416  -0.17060  -0.00056
2.5    0.40  2.74  0.78035  -0.17792  -0.00050
3      0.32  2.55  0.81112  -0.16625  -0.00044
4      0.25  2.36  0.84583  -0.15306  -0.00048
5      0.20  2.16  0.87177  -0.14444  -0.00052
6      0.16  1.98  0.89009  -0.13157  -0.00064
7.5    0.13  1.80  0.90635  -0.11886  -0.00081
10     0.10  1.62  0.91212  -0.10486  -0.00092
""")


class MotazedianAtkinson2005USGS(MotazedianAtkinson2005):
    """
    Modification of base class to include site response term used in the USGS NSHM.
    Gail Atkinson recommended use of site term from Boore and Atkinson (2008),
    which comes from Atkinson and Boore (2006).
    """
    kind = "usgs"

    #: Required rupture parameter is magnitude.
    #: Boore and Atkinson 2008 requires rake.
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'rake'}

    #: Required distance measure is Rrup.
    #: Boore and Atkinson 2008 requires Rjb.
    REQUIRES_DISTANCES = {'rrup', 'rjb' }

    #: Required site parameters is Vs30.
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        vars(ctx).update(contexts.get_dists(ctx))

        COEFFS_BA08 = BooreAtkinson2008.COEFFS

        # compute PGA on rock conditions - needed to compute non-linear
        # site amplification term
        pga4nl = _get_pga_on_rock(COEFFS_BA08[PGA()], ctx)
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]

            # Equation 7, page 13.
            R = _compute_r(ctx.rrup, ctx.mag)
            mean[m] = _compute_magnitude_scaling(ctx.mag, C) + \
                _compute_hinge_function(ctx.mag, R) + \
                _compute_anelastic_attenuation(C, R)
            mean[m] *= BASE_10_TO_E
            mean[m] -= LN_CM_TO_G

            # Add site response from Boore and Atkinson, 2006
            COEFFS_SOIL = AtkinsonBoore2006.COEFFS_SOIL_RESPONSE[imt]
            mean[m] += _get_site_amplification_linear(ctx.vs30, COEFFS_SOIL) + \
                    _get_site_amplification_non_linear(ctx.vs30, pga4nl, COEFFS_SOIL)

            sig[m] = 0.28 * BASE_10_TO_E
