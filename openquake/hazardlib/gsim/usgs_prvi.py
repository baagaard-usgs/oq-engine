# -*- coding: utf-8 -*-
"""
USGS adjustments to ground-motion models for Puerto Rico and the Virgin Islands.

The adjustments correspond to the fixed effect computed using linear mixed effects with
grouping for the event and site terms.
"""
import pathlib

from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib import const

from openquake.hazardlib.gsim import (
    abrahamson_2014,
    boore_2014,
    campbell_bozorgnia_2014,
    chiou_youngs_2014,
    cauzzi_2014,
    abrahamson_gulerce_2022,
    kuehn_2020,
    parker_2020,
    )


class AbrahamsonEtAl2014_USGSPRVI(abrahamson_2014.AbrahamsonEtAl2014):
    """
    Abrahamson et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "AbrahamsonEtAl2014_USGSPRVI_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class BooreEtAl2014_USGSPRVI(boore_2014.BooreEtAl2014):
    """
    Boore et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "BooreEtAl2014_USGSPRVI_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class CampbellBozorgnia2014_USGSPRVI(campbell_bozorgnia_2014.CampbellBozorgnia2014):
    """
    Campbell and Bozorgnia (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "CampbellBozorgnia2014_USGSPRVI_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class ChiouYoungs2014_USGSPRVI(chiou_youngs_2014.ChiouYoungs2014):
    """
    Chiou and Youngs (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "ChiouYoungs2014_USGSPRVI_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class CauzziEtAl2014_USGSPRVI(cauzzi_2014.CauzziEtAl2014):
    """
    Cauzzi et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "CauzziEtAl2014_USGSPRVI_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class AbrahamsonGulerce2022SInter_USGSPRVI(abrahamson_gulerce_2022.AbrahamsonGulerce2022SInter):
    """
    Abrahamson and Gulerce (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "AbrahamsonGulerce2022_USGSPRVI_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class AbrahamsonGulerce2022SSlab_USGSPRVI(AbrahamsonGulerce2022SInter_USGSPRVI):
    """
    Implements the 2020 subduction intraslab ground motion model of Abrahamson &
    Gulerce (2020) GMM without site response.
    """
    #: Required rupture parameters are magnitude and top-of-rupture depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'ztor'}

    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class KuehnEtAl2020SInter_USGSPRVI(kuehn_2020.KuehnEtAl2020SInter):
    """
    Kuehn et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "KuehnEtAl2020_USGSPRVI_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class KuehnEtAl2020SSlab_USGSPRVI(KuehnEtAl2020SInter_USGSPRVI):
    """
    Implements the 2020 subduction intraslab ground motion model of Kuehn
    et al. (2020) GMM without site response.
    """
    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class ParkerEtAl2020SInter_USGSPRVI(parker_2020.ParkerEtAl2020SInter):
    """
    Parker et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "ParkerEtAl2020_USGSPRVI_coeffs.csv"
    with open(filename, encoding="utf-8") as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())
    

class ParkerEtAl2020SSlab_USGSPRVI(ParkerEtAl2020SInter_USGSPRVI):
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

