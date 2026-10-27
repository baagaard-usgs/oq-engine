# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2018 GEM Foundation
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
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`KuehnEtAl2022SInter`,
               :class:`KuehnEtAl2022SInterAlaska`,
               :class:`KuehnEtAl2022SInterCascadia`,
               :class:`KuehnEtAl2022SInterCentralAmericaMexico`,
               :class:`KuehnEtAl2022SInterJapan`,
               :class:`KuehnEtAl2022SInterNewZealand`,
               :class:`KuehnEtAl2022SInterSouthAmerica`,
               :class:`KuehnEtAl2022SInterTaiwan`,
               :class:`KuehnEtAl2022SSlab`,
               :class:`KuehnEtAl2022SSlabAlaska`,
               :class:`KuehnEtAl2022SSlabCascadia`
               :class:`KuehnEtAl2022SSlabCentralAmericaMexico`,
               :class:`KuehnEtAl2022SSlabJapan`,
               :class:`KuehnEtAl2022SSlabNewZealand`,
               :class:`KuehnEtAl2022SSlabSouthAmerica`,
               :class:`KuehnEtAl2022SSlabTaiwan`
"""
import pathlib
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable, add_alias
from openquake.hazardlib.gsim import kuehn_2020


# Coefficients from GitHub, https://github.com/nikuehn/KBCG20/tree/master/COEFFS_SEP21
# The directory COEFFS_SEP21 contains new sets of coefficients based on updated Alaska data.
# Only the regional coefficients for Alaska have changed.
# https://github.com/nikuehn/KBCG20/commit/5d30338436b897822b411553c6ae0bf31fac5ce5
coeffs_filepath = pathlib.Path(__file__).parent / "kuehn_2022_coeffs.csv"


class KuehnEtAl2022SInter(kuehn_2020.KuehnEtAl2020SInter):
    """
    Updates KuehnEtAl2020SInter coefficients based on updated Alaska data.
    The global coefficients and coefficients for other regions have not been changed.
    """
    with open(coeffs_filepath, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class KuehnEtAl2022SSlab(kuehn_2020.KuehnEtAl2020SSlab):
    """
    Updates KuehnEtAl2020SSlab coefficients based on updated Alaska data.
    The global coefficients and coefficients for other regions have not been changed.
    """
    with open(coeffs_filepath, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())



for region in kuehn_2020.SUPPORTED_REGIONS[1:]:
    add_alias("KuehnEtAl2020SInter" + kuehn_2020.REGION_ALIASES[region],
              KuehnEtAl2022SInter, region=region)
    add_alias("KuehnEtAl2020SSlab" + kuehn_2020.REGION_ALIASES[region],
              KuehnEtAl2022SSlab, region=region)
