"""
Class label definitions and mappings for the CCTV AI Fight Detection system.
"""

# Binary class mapping
# 0 = Non-Fight
# 1 = Fight
LABEL_NON_FIGHT = 0
LABEL_FIGHT = 1

ID_TO_LABEL = {
    LABEL_NON_FIGHT: "Non-Fight",
    LABEL_FIGHT: "Fight"
}

LABEL_TO_ID = {
    "Non-Fight": LABEL_NON_FIGHT,
    "Fight": LABEL_FIGHT,
    "non_fight": LABEL_NON_FIGHT,
    "nonfight": LABEL_NON_FIGHT,
    "fight": LABEL_FIGHT,
    "normal": LABEL_NON_FIGHT
}

NUM_CLASSES = 2

