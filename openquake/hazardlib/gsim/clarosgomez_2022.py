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
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


def _compute_r(rhypo, h):
    return np.sqrt(rhypo**2 + h**2)


def _compute_magnitude_scaling(mag, C):
    return C["b1"] + C["b2"]*mag


def _compute_attenuation(r, C):
    return C["b3"]*np.log(r) + C["b4"]*r


def _compute_site_term(vs30, C):
    res = np.zeros_like(vs30)

    idx = np.logical_and(vs30 > 300.0, vs30 <= 600.0)
    res[idx] = C["b6"]

    idx = vs30 <= 300.0
    res[idx] = C["b5"]

    return res

class ClarosGomez2022(GMPE):
    """
    Implements GMPE developed by Diego Fernando Claros Gómez and published as
    "Ground Motion Prediction Equation for Puerto Rico", Diego Fernando Claros
    Gomez, University of Puerto Rico, Mayaguez Campus, Ph.D. Thesis, 2022.

    ln Y = b1 + b2*M + b3*ln(sqrt(d**2+h**2)) + b4*sqrt(d**2+h**2) + b5*S + b6*H
    """
    #: Supported intensity measure types are spectral acceleration,
    #: peak ground velocity and peak ground acceleration, see Table 4.2
    #: pg 43.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    #: Supported intensity measure component is orientation-independent
    #: measure :attr:`~openquake.hazardlib.const.IMC.GMRotI50`.
    #: Component not mentioned in manuscript, but use of gmprocess for ground-
    #: motion processing strongly suggests RotD50.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GMRotI50

    #: Supported standard deviation types are total.
    #: See section 3.5.3 Linear regression in two stages, pg 39.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = { const.StdDev.TOTAL }

    #: Required site parameters is Vs30.
    #: See Table 3.8, pg 37.
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameter is magnitude.
    #: See section 3.5.1 Functional form, pg 38.
    REQUIRES_RUPTURE_PARAMETERS = {'mag'}

    #: Required distance measure is Rhypo.
    #: See section 3.5.1 Functional form, pg 38.
    REQUIRES_DISTANCES = {'rhypo'}

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

            r = _compute_r(ctx.rhypo, self.H)
            mean[m] = _compute_magnitude_scaling(ctx.mag, C) + \
                _compute_attenuation(r, C) + \
                    _compute_site_term(ctx.vs30, C)

            sig[m] = C["sigma"]


class ClarosGomez2022Crustal(ClarosGomez2022):
    """
    Implementation of ClarosGomez2022 for crustal earthquakes.
    """
    H = 2.3

    #: Supported tectonic region type is active shallow crust, see
    #: section 3.1.1 Crustal data, pg 20.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST
    
    COEFFS = CoeffsTable(sa_damping=5, table="""\
IMT          b1     b2      b3      b4      b5      b6  sigma
   PGA  -7.0579 1.7277 -1.6702 -0.0039  0.1731  0.1206  0.854
   PGV  -5.2371 2.0359 -1.6298 -0.0028  0.4998  0.3175  0.813
 0.010  -6.9916 1.7190 -1.6731 -0.0038  0.1743  0.1178  0.852
 0.020  -6.9275 1.7202 -1.6845 -0.0038  0.1636  0.1107  0.855
 0.030  -6.6811 1.7159 -1.7244 -0.0036  0.1182  0.0856  0.869
 0.050  -6.1697 1.6906 -1.7423 -0.0041 -0.0548 -0.0320  0.889
 0.075  -5.4875 1.6537 -1.7373 -0.0047 -0.3148 -0.2151  0.928
 0.100  -5.5614 1.6563 -1.6903 -0.0050 -0.2212 -0.0980  0.927
 0.150  -6.1636 1.6555 -1.5572 -0.0046  0.1856  0.0909  0.871
 0.200  -6.8735 1.6827 -1.4678 -0.0046  0.4732  0.4003  0.872
 0.250  -7.4898 1.7772 -1.4834 -0.0043  0.6238  0.4762  0.889
 0.300  -7.9041 1.8526 -1.5477 -0.0033  0.7688  0.4940  0.882
 0.400  -8.9440 1.9918 -1.5486 -0.0026  0.7088  0.4078  0.871
 0.500  -9.6468 2.0993 -1.6085 -0.0017  0.6946  0.3842  0.863
 0.750 -11.2119 2.3188 -1.6584 -0.0004  0.6377  0.2938  0.866
 1.000 -12.3880 2.4669 -1.6832  0.0002  0.5866  0.3055  0.875
 1.500 -13.9531 2.6489 -1.6965  0.0000  0.6114  0.3591  0.902
 2.000 -14.9605 2.7256 -1.6540 -0.0008  0.6115  0.3561  0.893
 3.000 -16.0692 2.7714 -1.5983 -0.0017  0.5809  0.3589  0.877
 4.000 -16.6258 2.7709 -1.5814 -0.0021  0.5782  0.3612  0.851
 5.000 -17.0482 2.7611 -1.5500 -0.0024  0.5611  0.3508  0.845
 7.500 -17.2077 2.6477 -1.5411 -0.0025  0.5307  0.3291  0.819
10.000 -17.0478 2.5296 -1.5587 -0.0024  0.5074  0.3239  0.811
""")

class ClarosGomez2022Noncrustal(ClarosGomez2022):
    """
    Implementation of ClarosGomez2022 for non-crustal earthquakes.
    Most of these are intraslab earthquakes.
    """
    H = 44.0

    #: Supported tectonic region type is subduction intraslab, see
    #: section 3.1.2 Non-crustal data, pg 22.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB
    
    COEFFS = CoeffsTable(sa_damping=5, table="""\
IMT         b1     b2      b3     b4      b5      b6  sigma
   PGA  3.8241 2.7096 -4.9726 0.0105  0.0876 -0.0419  0.969
   PGV  3.6779 2.9251 -4.3717 0.0085  0.1778  0.0369  0.933
 0.010  3.9279 2.6970 -4.9808 0.0106  0.1160 -0.0369  0.968
 0.020  4.0532 2.6954 -5.0023 0.0107  0.1034 -0.0453  0.971
 0.030  4.4088 2.6925 -5.0661 0.0109  0.0654 -0.0700  0.984
 0.050  5.3546 2.7004 -5.2294 0.0112 -0.0409 -0.1317  1.018
 0.075  7.8567 2.7042 -5.6576 0.0125 -0.3845 -0.3475  1.084
 0.100  7.8211 2.7384 -5.6288 0.0122 -0.4099 -0.3364  1.072
 0.150  5.8108 2.6364 -5.0944 0.0110  0.1229 -0.1358  0.994
 0.200  2.7245 2.6478 -4.4816 0.0088  0.4949  0.2650  0.983
 0.250  1.3884 2.7609 -4.3600 0.0087  0.5338  0.2835  0.959
 0.300 -0.2008 2.8325 -4.1104 0.0076  0.5610  0.2013  0.948
 0.400 -2.9009 2.9136 -3.6650 0.0059  0.4023  0.0704  0.963
 0.500 -4.6255 3.0447 -3.4923 0.0055  0.3175 -0.0180  0.958
 0.750 -6.4255 3.3010 -3.5097 0.0059  0.2375 -0.1233  1.009
 1.000 -7.6759 3.5096 -3.5548 0.0060  0.2106 -0.1174  1.070
 1.500 -8.6049 3.7083 -3.7075 0.0061  0.2161 -0.1097  1.079
 2.000 -8.6942 3.7093 -3.8311 0.0064  0.2212 -0.0616  1.044
 3.000 -8.6305 3.6020 -3.9324 0.0067  0.2239 -0.0066  1.010
 4.000 -8.2496 3.4676 -4.0131 0.0070  0.2373  0.0140  0.985
 5.000 -8.0679 3.3735 -4.0495 0.0071  0.2381  0.0193  0.969
 7.500 -7.8574 3.2431 -4.1075 0.0073  0.2185  0.0239  0.958
10.000 -7.6880 3.1556 -4.1518 0.0075  0.2055  0.0250  0.950
""")

