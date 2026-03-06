# -*- coding: utf-8 -*-
"""
USGS adjustments to ground-motion models for Samoa and the Northern Mariana Islands.


"""
import pathlib

from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib import const

from openquake.hazardlib.gsim import (
    abrahamson_gulerce_2022,
    kuehn_2020,
    parker_2020,
    )


class AbrahamsonGulerce2022SInter_USGSSamoa(abrahamson_gulerce_2022.AbrahamsonGulerce2022SInter):
    """
    Abrahamson and Gulerce (2020) subduction interface ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    filename = pathlib.Path(__file__).parent / "AbrahamsonGulerce2022_USGSSamoa_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class AbrahamsonGulerce2022SSlab_USGSSamoa(AbrahamsonGulerce2022SInter_USGSSamoa):
    """
    Abrahamson and Gulerce (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    #: Required rupture parameters are magnitude and top-of-rupture depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'ztor'}

    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class KuehnEtAl2020SInter_USGSSamoa(kuehn_2020.KuehnEtAl2020SInter):
    """
    Kuehn et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    filename = pathlib.Path(__file__).parent / "KuehnEtAl2020_USGSSamoa_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class KuehnEtAl2020SSlab_USGSSamoa(KuehnEtAl2020SInter_USGSSamoa):
    """
    Kuehn et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class ParkerEtAl2020SInter_USGSSamoa(parker_2020.ParkerEtAl2020SInter):
    """
    Parker et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "ParkerEtAl2020_USGSSamoa_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())
    

class ParkerEtAl2020SSlab_USGSSamoa(ParkerEtAl2020SInter_USGSSamoa):
    """
    Modifications for subduction slab.
    """
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB

    # slab also requires hypo_depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'hypo_depth'}

    # constant table suffix
    SUFFIX = "slab"

    MB_REGIONS = {"Aleutian": 7.98, "AK": 7.2, "Cascadia": 7.2,
                  "CAM_S": 7.6, "CAM_N": 7.4, "JP_Pac": 7.65, "JP_Phi": 7.55,
                  "SA_N": 7.3, "SA_S": 7.25, "TW_W": 7.7, "TW_E": 7.7,
                  "default": 7.6}

