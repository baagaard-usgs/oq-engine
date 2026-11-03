# -*- coding: utf-8 -*-
"""
USGS adjustments to ground-motion models for American Samoa.


"""
import pathlib

from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib import const

from openquake.hazardlib.gsim import (
    abrahamson_gulerce_2022,
    kuehn_2022,
    parker_2022,
    )


class AbrahamsonGulerce2022SInter_USGSAmSamStage1(abrahamson_gulerce_2022.AbrahamsonGulerce2022SInter):
    """
    Abrahamson and Gulerce (2022) subduction interface ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    filename = pathlib.Path(__file__).parent / "AbrahamsonGulerce2022_USGSAmSamStage1_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class AbrahamsonGulerce2022SSlab_USGSAmSamStage1(AbrahamsonGulerce2022SInter_USGSAmSamStage1):
    """
    Abrahamson and Gulerce (2022) subduction intraslab ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    #: Required rupture parameters are magnitude and top-of-rupture depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'ztor'}

    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class KuehnEtAl2022SInter_USGSAmSamStage1(kuehn_2022.KuehnEtAl2022SInter):
    """
    Kuehn et al. (2022) subduction interface ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    filename = pathlib.Path(__file__).parent / "KuehnEtAl2022_USGSAmSamStage1_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class KuehnEtAl2022SSlab_USGSAmSamStage1(KuehnEtAl2022SInter_USGSAmSamStage1):
    """
    Kuehn et al. (2022) subduction intraslab ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class ParkerEtAl2022SInter_USGSAmSamStage1(parker_2022.ParkerEtAl2022SInter):
    """
    Parker et al. (2022) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "ParkerEtAl2022_USGSAmSamStage1_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())
    

class ParkerEtAl2022SSlab_USGSAmSamStage1(ParkerEtAl2022SInter_USGSAmSamStage1):
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

class AbrahamsonGulerce2022SInter_USGSAmSamStage2(abrahamson_gulerce_2022.AbrahamsonGulerce2022SInter):
    """
    Abrahamson and Gulerce (2022) subduction interface ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    filename = pathlib.Path(__file__).parent / "AbrahamsonGulerce2022_USGSAmSamStage2_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class AbrahamsonGulerce2022SSlab_USGSAmSamStage2(AbrahamsonGulerce2022SInter_USGSAmSamStage2):
    """
    Abrahamson and Gulerce (2022) subduction intraslab ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    #: Required rupture parameters are magnitude and top-of-rupture depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'ztor'}

    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class KuehnEtAl2022SInter_USGSAmSamStage2(kuehn_2022.KuehnEtAl2022SInter):
    """
    Kuehn et al. (2022) subduction interface ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    filename = pathlib.Path(__file__).parent / "KuehnEtAl2022_USGSAmSamStage2_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class KuehnEtAl2022SSlab_USGSAmSamStage2(KuehnEtAl2022SInter_USGSAmSamStage2):
    """
    Kuehn et al. (2022) subduction intraslab ground-motion model with 
    USGS adjustments for Samoa and the Northern Mariana Islands.
    """
    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class ParkerEtAl2022SInter_USGSAmSamStage2(parker_2022.ParkerEtAl2022SInter):
    """
    Parker et al. (2022) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "ParkerEtAl2022_USGSAmSamStage2_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())
    

class ParkerEtAl2022SSlab_USGSAmSamStage2(ParkerEtAl2022SInter_USGSAmSamStage2):
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

