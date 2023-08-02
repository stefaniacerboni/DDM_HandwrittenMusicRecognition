rhythm_mapping = {
    0: 'epsilon',
    1: 'barline_light',
    2: 'barline_light-light',
    3: 'beam8thDown',
    4: 'beam8thUp',
    5: 'beamDownEnd',
    6: 'beamDownStart',
    7: 'beamUpEnd',
    8: 'beamUpStart',
    9: 'C-Clef',
    10: 'dot',
    11: 'eighthRest',
    12: 'endSlur',
    13: 'flag16thDown',
    14: 'flag8thDown',
    15: 'flag8thUp',
    16: 'flat',
    17: 'halfRest',
    18: 'mmrSymbol_1',
    19: 'mmrSymbol_2',
    20: 'mmrSymbol_3',
    21: 'natural',
    22: 'noteheadBlack',
    23: 'noteheadHalf',
    24: 'noteheadWhole',
    25: 'quarterRest',
    26: 'sharp',
    27: 'startSlur',
    28: 'steamQuarterHalfDown',
    29: 'steamQuarterHalfUp',
    30: 'timeSig_common',
    31: '32thRest'
}
inverse_rhythm_mapping = {value: key for key, value in rhythm_mapping.items()}

pitch_mapping = {
    0: 'epsilon',
    1: 'noNote',
    2: 'L1',
    3: 'L2',
    4: 'L3',
    5: 'L4',
    6: 'L5',
    7: 'L6',
    8: 'S1',
    9: 'S2',
    10: 'S3',
    11: 'S4',
    12: 'S5',
    13: 'S6'
}

inverse_pitch_mapping = {value: key for key, value in pitch_mapping.items()}

