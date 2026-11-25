import numpy as np

from matplotlib.colors import Normalize, PowerNorm, LinearSegmentedColormap

# heading tiers
H1 = np.array([0.15, 0.15, 0.15, 1.0])
H2 = np.array([0.30, 0.30, 0.30, 1.0])
H3 = np.array([0.45, 0.45, 0.45, 1.0])

# make highlights
HL1 = np.array([0.35, 0.35, 0.35, 1.0])
HL2 = np.array([0.50, 0.50, 0.50, 1.0])
HL3 = np.array([0.00, 0.75, 0.00, 1.0])

CM_HL = LinearSegmentedColormap.from_list(
    'highlights',
    [HL1, HL2, HL3]
)

NM_HL = PowerNorm(
    gamma = 0.5,
    vmin = 0,
    vmax = 1
)

HL_EMBEDDING = CM_HL(NM_HL(0.0))
HL_OTHERS = CM_HL(NM_HL(0.25))
HL_SELF = CM_HL(NM_HL(0.75))

# make rdms
RDM1 = np.array([0.4, 0.4, 0.4, 1.0])
RDM2 = np.array([0.95, 0.95, 0.95, 1.0])

CM_RDM_CL = LinearSegmentedColormap.from_list(
    'rdm_cl',
    [RDM1, RDM2]
)

CM_RDM_HL = LinearSegmentedColormap.from_list(
    'rdm_hl',
    [RDM1, HL_SELF]
)