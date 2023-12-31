# Chiave di interpretazione umana per gli indici del rhythm_mapping:
# Indice a una cifra: simbolo semplice
# Indici a più cifre:
# 1a cifra: 2 per le pause, 1 per le note
# 2a cifra (solo note): 0 = normale, 5 = dotted (durata +50%)
# 3a cifra (solo note): 0 = normale, 1 = flat, 2 = sharp, 3 = natural (cioè flat o sharp naturalizzata singolarmente)
# Ultime 3 cifre indicano la durata: 132 (un trentaduesimo), 116 (un sedicesimo), 201 (due battute), eccetera

# Rhythm mapping
human_readable_rhythm_mapping = {
    0 : 'blank',
    1 : 'epsilon',
    2 : 'barline',
    3 : 'C-Clef',
    4 : 'startSlur',  # inizio legatura
    5 : 'endSlur',  # fine legatura
    6 : 'timeSig_common',

    2132 : '32thRest',
    2116 : '16thRest',
    2108 : 'eighthRest',
    2104 : 'quarterRest',
    2102 : 'halfRest',
    2101 : 'wholeRest', # mmrSymbol_1 -> pausa da 4/4
    2201 : 'doubleWholeRest', # mmrSymbol_2 -> pausa da 8/4
    2401 : 'quadrupleWholeRest', # mmrSymbol_3 -> pausa da 16/4

    100132 : '32thNote',
    100116 : '16thNote',
    100108 : 'eighthNote',
    100104 : 'quarterNote',
    100102 : 'halfNote',
    100101 : 'wholeNote',

    150132 : 'dotted32thNote',
    150116 : 'dotted16thNote',
    150108 : 'dottedEighthNote',
    150104 : 'dottedQuarterNote',
    150102 : 'dottedHalfNote',
    150101 : 'dottedWholeNote',

    101132 : 'flat32thNote',
    101116 : 'flat16thNote',
    101108 : 'flatEighthNote',
    101104 : 'flatQuarterNote',
    101102 : 'flatHalfNote',
    101101 : 'flatWholeNote',

    102132 : 'sharp32thNote',
    102116 : 'sharp16thNote',
    102108 : 'sharpEighthNote',
    102104 : 'sharpQuarterNote',
    102102 : 'sharpHalfNote',
    102101 : 'sharpWholeNote',

    103132 : 'natural32thNote',
    103116 : 'natural16thNote',
    103108 : 'naturalEighthNote',
    103104 : 'naturalQuarterNote',
    103102 : 'naturalHalfNote',
    103101 : 'naturalWholeNote',

    151132 : 'dottedFlat32thNote',
    151116 : 'dottedFlat16thNote',
    151108 : 'dottedFlatEighthNote',
    151104 : 'dottedFlatQuarterNote',
    151102 : 'dottedFlatHalfNote',
    151101 : 'dottedFlatWholeNote',

    152132 : 'dottedSharp32thNote',
    152116 : 'dottedSharp16thNote',
    152108 : 'dottedSharpEighthNote',
    152104 : 'dottedSharpQuarterNote',
    152102 : 'dottedSharpHalfNote',
    152101 : 'dottedSharpWholeNote',

    153132 : 'dottedNatural32thNote',
    153116 : 'dottedNatural16thNote',
    153108 : 'dottedNaturalEighthNote',
    153104 : 'dottedNaturalQuarterNote',
    153102 : 'dottedNaturalHalfNote',
    153101 : 'dottedNaturalWholeNote',
}


# Chiave di interpretazione umana per gli indici del pitch_mapping:
# Indice a una cifra: simbolo semplice
# Indici a due cifre:
# 1a cifra: 4 per le Lines, 5 per gli Spaces
# 2a cifra: la linea o lo spazio

# Hard-coded modified pitch mapping
human_readable_pitch_mapping = {
    0 : 'blank',
    1 : 'epsilon',
    2 : 'noNote',

    40 : 'L0',
    41 : 'L1',
    42 : 'L2',
    43 : 'L3',
    44 : 'L4',
    45 : 'L5',
    46 : 'L6',

    50 : 'S0',
    51 : 'S1',
    52 : 'S2',
    53 : 'S3',
    54 : 'S4',
    55 : 'S5',
    56 : 'S6'
}

rhythm_mapping = {
    0: 'blank',
    1: 'epsilon',
    2: 'barline',
    3: 'C-Clef',
    4: 'startSlur',
    5: 'endSlur',
    6: 'timeSig_common',
    7: '32thRest',
    8: '16thRest',
    9: 'eighthRest',
    10: 'quarterRest',
    11: 'halfRest',
    12: 'wholeRest',
    13: 'doubleWholeRest',
    14: 'quadrupleWholeRest',
    15: '32thNote',
    16: '16thNote',
    17: 'eighthNote',
    18: 'quarterNote',
    19: 'halfNote',
    20: 'wholeNote',
    21: 'dotted32thNote',
    22: 'dotted16thNote',
    23: 'dottedEighthNote',
    24: 'dottedQuarterNote',
    25: 'dottedHalfNote',
    26: 'dottedWholeNote',
    27: 'flat32thNote',
    28: 'flat16thNote',
    29: 'flatEighthNote',
    30: 'flatQuarterNote',
    31: 'flatHalfNote',
    32: 'flatWholeNote',
    33: 'sharp32thNote',
    34: 'sharp16thNote',
    35: 'sharpEighthNote',
    36: 'sharpQuarterNote',
    37: 'sharpHalfNote',
    38: 'sharpWholeNote',
    39: 'natural32thNote',
    40: 'natural16thNote',
    41: 'naturalEighthNote',
    42: 'naturalQuarterNote',
    43: 'naturalHalfNote',
    44: 'naturalWholeNote',
    45: 'dottedFlat32thNote',
    46: 'dottedFlat16thNote',
    47: 'dottedFlatEighthNote',
    48: 'dottedFlatQuarterNote',
    49: 'dottedFlatHalfNote',
    50: 'dottedFlatWholeNote',
    51: 'dottedSharp32thNote',
    52: 'dottedSharp16thNote',
    53: 'dottedSharpEighthNote',
    54: 'dottedSharpQuarterNote',
    55: 'dottedSharpHalfNote',
    56: 'dottedSharpWholeNote',
    57: 'dottedNatural32thNote',
    58: 'dottedNatural16thNote',
    59: 'dottedNaturalEighthNote',
    60: 'dottedNaturalQuarterNote',
    61: 'dottedNaturalHalfNote',
    62: 'dottedNaturalWholeNote'
}

pitch_mapping = {
    0: 'blank',
    1: 'epsilon',
    2: 'noNote',
    3: 'L0',
    4: 'L1',
    5: 'L2',
    6: 'L3',
    7: 'L4',
    8: 'L5',
    9: 'L6',
    10: 'S0',
    11: 'S1',
    12: 'S2',
    13: 'S3',
    14: 'S4',
    15: 'S5',
    16: 'S6'
}

inverse_rhythm_mapping = {
    'blank': 0,
    'epsilon': 1,
    'barline': 2,
    'C-Clef': 3,
    'startSlur': 4,
    'endSlur': 5,
    'timeSig_common': 6,
    '32thRest': 7,
    '16thRest': 8,
    'eighthRest': 9,
    'quarterRest': 10,
    'halfRest': 11,
    'wholeRest': 12,
    'doubleWholeRest': 13,
    'quadrupleWholeRest': 14,
    '32thNote': 15,
    '16thNote': 16,
    'eighthNote': 17,
    'quarterNote': 18,
    'halfNote': 19,
    'wholeNote': 20,
    'dotted32thNote': 21,
    'dotted16thNote': 22,
    'dottedEighthNote': 23,
    'dottedQuarterNote': 24,
    'dottedHalfNote': 25,
    'dottedWholeNote': 26,
    'flat32thNote': 27,
    'flat16thNote': 28,
    'flatEighthNote': 29,
    'flatQuarterNote': 30,
    'flatHalfNote': 31,
    'flatWholeNote': 32,
    'sharp32thNote': 33,
    'sharp16thNote': 34,
    'sharpEighthNote': 35,
    'sharpQuarterNote': 36,
    'sharpHalfNote': 37,
    'sharpWholeNote': 38,
    'natural32thNote': 39,
    'natural16thNote': 40,
    'naturalEighthNote': 41,
    'naturalQuarterNote': 42,
    'naturalHalfNote': 43,
    'naturalWholeNote': 44,
    'dottedFlat32thNote': 45,
    'dottedFlat16thNote': 46,
    'dottedFlatEighthNote': 47,
    'dottedFlatQuarterNote': 48,
    'dottedFlatHalfNote': 49,
    'dottedFlatWholeNote': 50,
    'dottedSharp32thNote': 51,
    'dottedSharp16thNote': 52,
    'dottedSharpEighthNote': 53,
    'dottedSharpQuarterNote': 54,
    'dottedSharpHalfNote': 55,
    'dottedSharpWholeNote': 56,
    'dottedNatural32thNote': 57,
    'dottedNatural16thNote': 58,
    'dottedNaturalEighthNote': 59,
    'dottedNaturalQuarterNote': 60,
    'dottedNaturalHalfNote': 61,
    'dottedNaturalWholeNote': 62
}

inverse_pitch_mapping = {
    'blank': 0,
    'epsilon': 1,
    'noNote': 2,
    'L0': 3,
    'L1': 4,
    'L2': 5,
    'L3': 6,
    'L4': 7,
    'L5': 8,
    'L6': 9,
    'S0': 10,
    'S1': 11,
    'S2': 12,
    'S3': 13,
    'S4': 14,
    'S5': 15,
    'S6': 16
}

inverse_human_readable_rhythm_mapping = {
    'blank': 0,
    'epsilon': 1,
    'barline': 2,
    'C-Clef': 3,
    'startSlur': 4,
    'endSlur': 5,
    'timeSig_common': 6,
    '32thRest': 2132,
    '16thRest': 2116,
    'eighthRest': 2108,
    'quarterRest': 2104,
    'halfRest': 2102,
    'wholeRest': 2101,
    'doubleWholeRest': 2201,
    'quadrupleWholeRest': 2401,
    '32thNote': 100132,
    '16thNote': 100116,
    'eighthNote': 100108,
    'quarterNote': 100104,
    'halfNote': 100102,
    'wholeNote': 100101,
    'dotted32thNote': 105132,
    'dotted16thNote': 105116,
    'dottedEighthNote': 105108,
    'dottedQuarterNote': 105104,
    'dottedHalfNote': 105102,
    'dottedWholeNote': 105101,
    'flat32thNote': 110132,
    'flat16thNote': 110116,
    'flatEighthNote': 110108,
    'flatQuarterNote': 110104,
    'flatHalfNote': 110102,
    'flatWholeNote': 110101,
    'sharp32thNote': 120132,
    'sharp16thNote': 120116,
    'sharpEighthNote': 120108,
    'sharpQuarterNote': 120104,
    'sharpHalfNote': 120102,
    'sharpWholeNote': 120101,
    'natural32thNote': 130132,
    'natural16thNote': 130116,
    'naturalEighthNote': 130108,
    'naturalQuarterNote': 130104,
    'naturalHalfNote': 130102,
    'naturalWholeNote': 130101,
    'dottedFlat32thNote': 151132,
    'dottedFlat16thNote': 151116,
    'dottedFlatEighthNote': 151108,
    'dottedFlatQuarterNote': 151104,
    'dottedFlatHalfNote': 151102,
    'dottedFlatWholeNote': 151101,
    'dottedSharp32thNote': 152132,
    'dottedSharp16thNote': 152116,
    'dottedSharpEighthNote': 152108,
    'dottedSharpQuarterNote': 152104,
    'dottedSharpHalfNote': 152102,
    'dottedSharpWholeNote': 152101,
    'dottedNatural32thNote': 153132,
    'dottedNatural16thNote': 153116,
    'dottedNaturalEighthNote': 153108,
    'dottedNaturalQuarterNote': 153104,
    'dottedNaturalHalfNote': 153102,
    'dottedNaturalWholeNote': 153101
}

inverse_human_readable_pitch_mapping = {
    'blank': 0,
    'epsilon': 1,
    'noNote': 2,
    'L0': 40,
    'L1': 41,
    'L2': 42,
    'L3': 43,
    'L4': 44,
    'L5': 45,
    'L6': 46,
    'S0': 50,
    'S1': 51,
    'S2': 52,
    'S3': 53,
    'S4': 54,
    'S5': 55,
    'S6': 56
}
'''
# Create a new mapping with incremental values
new_mapping = {}
current_value = 0

# Iterate through inverse_rhythm_mapping and apply the specified rules
for key, value in inverse_rhythm_mapping.items():
    if key == 'blank':
        continue
    elif key == 'epsilon':
        new_mapping[key] = current_value
        current_value += 1
    elif key in ['barline', 'startSlur', 'endSlur', 'timeSig_common']:
        new_mapping[f'{key}.noNote'] = current_value
        current_value += 1
    elif key.endswith('Rest'):
        new_mapping[f'{key}.noNote'] = current_value
        current_value += 1
    elif key == 'C-Clef':
        for i in range(1, 6):
            new_mapping[f'{key}.L{i}'] = current_value
            current_value += 1
    elif key.endswith('Note'):
        for note_key in inverse_pitch_mapping.keys():
            if note_key.startswith('L') or note_key.startswith('S'):
                new_mapping[f'{key}.{note_key}'] = current_value
                current_value += 1
# Create the inverse mapping
inverse_mapping = {v: k for k, v in new_mapping.items()}
# Write the new mapping to a file
with open("../seq2seq-model/new_mapping_combined.py", "w") as file:
    file.write("new_mapping_combined = {\n")
    for key, value in inverse_mapping.items():
        file.write(f"    {key}: '{value}',\n")
    file.write("}\n")
    file.write("inverse_new_mapping_combined = {\n")
    for key, value in new_mapping.items():
        file.write(f"    '{key}': {value},\n")
    file.write("}\n")
print("Mapping has been written to 'new_mapping_combined.py'")'''