from new_mapping import human_readable_rhythm_mapping, human_readable_pitch_mapping

conversion_map = {
    "blank": "blank",
    "barline_light": "barline",
    "barline_light-light": "barline",
    "C-Clef": "C-Clef",
    "beam8thDown": "",
    "beam8thUp": "",
    "beamDownEnd.noNote~epsilon~": "",
    "~epsilon~beamDownStart.noNote": "",
    "beamUpEnd.noNote~epsilon~": "",
    "~epsilon~beamUpStart.noNote": "",
    "startSlur": "startSlur",
    "endSlur": "endSlur",
    "32thRest": "32thRest",
    "16thRest": "16thRest",
    "eighthRest": "eighthRest",
    "quarterRest": "quarterRest",
    "halfRest": "halfRest",
    "mmrSymbol_1": "wholeRest",
    "mmrSymbol_2": "doubleWholeRest",
    "mmrSymbol_3": "quadrupleWholeRest",
    # ...
}


# Chiave di interpretazione umana per gli indici del rhythm_mapping:
# Indice a una cifra: simbolo semplice
# Indici a più cifre:
# 1a cifra: 2 per le pause, 1 per le note
# 2a cifra (solo note): 0 = normale, 5 = dotted (durata +50%)
# 3a cifra (solo note): 0 = normale, 1 = flat, 2 = sharp, 3 = natural (cioè flat o sharp naturalizzata singolarmente)
# Ultime 3 cifre indicano la durata: 132 (un trentaduesimo), 116 (un sedicesimo), 201 (due battute), eccetera
def generate_note_key(prefix, note_type, position, dotted=False):
    """Genera la chiave per una nota basata sulle regole fornite."""
    base = 100000  # inizia con una nota
    if prefix == "flat":
        base += 1000
    elif prefix == "sharp":
        base += 2000
    elif prefix == "natural":
        base += 3000
    if dotted:
        base += 50000
    if note_type == "32thNote":
        base += 132
    elif note_type == "16thNote":
        base += 116
    elif note_type == "eighthNote":
        base += 108
    elif note_type == "quarterNote":
        base += 104
    elif note_type == "halfNote":
        base += 102
    elif note_type == "wholeNote":
        base += 101
    return f"{human_readable_rhythm_mapping[base]}.{position}"


prefixes = ["", "flat", "sharp", "natural"]
note_types = ["32thNote", "16thNote", "eighthNote", "quarterNote", "halfNote", "wholeNote"]
positions = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "S0", "S1", "S2", "S3", "S4", "S5", "S6"]

# Generazione mappa di conversione
for prefix in prefixes:
    for note_type in note_types:
        for position in positions:
                for dotted in [False, True]:
                    old_keys = []
                    if prefix:
                        prefix_str = f"{prefix}.{position}~epsilon~"
                    else:
                        prefix_str = ""
                    if note_type == "32thNote":
                        old_keys.append(f"{prefix_str}noteheadBlack.{position}~flag16thDown.noNote")
                        old_keys.append(f"{prefix_str}flag16thUp.noNote~noteheadBlack.{position}")
                    elif note_type == "16thNote":
                        old_keys.append(f"{prefix_str}noteheadBlack.{position}~flag16thDown.noNote")
                        old_keys.append(f"{prefix_str}flag16thUp.noNote~noteheadBlack.{position}")
                    elif note_type == "eighthNote":
                        old_keys.append(f"{prefix_str}noteheadBlack.{position}~flag8thDown.noNote")
                        old_keys.append(f"{prefix_str}flag8thUp.noNote~noteheadBlack.{position}")
                        old_keys.append(f"{prefix_str}noteheadBlack.{position}~beam8thDown.noNote")
                        old_keys.append(f"{prefix_str}beam8thUp.noNote~noteheadBlack.{position}")
                    elif note_type == "quarterNote":
                        old_keys.append(f"{prefix_str}noteheadBlack.{position}~steamQuarterHalfDown.noNote")
                        old_keys.append(f"{prefix_str}steamQuarterHalfUp.noNote~noteheadBlack.{position}")
                    elif note_type == "halfNote":
                        old_keys.append(f"{prefix_str}noteheadHalf.{position}~steamQuarterHalfDown.noNote")
                        old_keys.append(f"{prefix_str}steamQuarterHalfUp.noNote~noteheadHalf.{position}")
                    elif note_type == "wholeNote":
                        old_keys.append(f"{prefix_str}noteheadWhole.{position}")
                    new_key = generate_note_key(prefix, note_type, position, dotted)

                    for old_key in old_keys:
                        if dotted:
                            old_key += "~epsilon~dot.noNote"
                        conversion_map[old_key] = new_key


# Funzione di conversione
def convert_classes(old_string):
    new_string = old_string
    # ordina per lunghezza per evitare sostituzioni parziali
    for old_class, new_class in sorted(conversion_map.items(), key=lambda x: len(x[0]), reverse=True):
        new_string = new_string.replace(old_class, new_class)
    return new_string


for old_key, new_key in conversion_map.items():
    print(f"{old_key} ---> {new_key}")

# Applica la conversione ai tre file di input e scrive i file con le nuove annotazioni
with open("../gt_final.valid.thresh", "r") as infile:
    lines = infile.readlines()
converted_lines = [convert_classes(line) for line in lines]
with open("newdef_gt_final.valid.thresh", "w") as outfile:
    outfile.writelines(converted_lines)

with open("../gt_final.train.thresh", "r") as infile:
    lines = infile.readlines()
converted_lines = [convert_classes(line) for line in lines]
with open("newdef_gt_final.train.thresh", "w") as outfile:
    outfile.writelines(converted_lines)

with open("../gt_final.test.thresh", "r") as infile:
    lines = infile.readlines()
converted_lines = [convert_classes(line) for line in lines]
with open("newdef_gt_final.test.thresh", "w") as outfile:
    outfile.writelines(converted_lines)