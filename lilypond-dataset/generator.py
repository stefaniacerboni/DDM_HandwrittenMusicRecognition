import random
import subprocess
import cv2
import os
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageOps
from tqdm import tqdm

from new_mapping import human_readable_rhythm_mapping, human_readable_pitch_mapping

header = '''\\version "2.24.2"
\\paper {
line-width = 210\\mm
oddFooterMarkup=##f
oddHeaderMarkup=##f
bookTitleMarkup = ##f
scoreTitleMarkup = ##f
ragged-bottom = ##t
ragged-last-bottom = ##t
ragged-right = ##t
footer = ##f
}
{
\\override Staff.Clef.stencil = ##f
\\override Staff.TimeSignature.stencil = ##f
'''

lilypond_cclef_mapping = {
    'C-Clef.L1': '\\revert Staff.Clef.stencil \\clef "soprano"',
    'C-Clef.L2': '\\revert Staff.Clef.stencil \\clef "mezzosoprano"',
    'C-Clef.L3': '\\revert Staff.Clef.stencil \\clef "alto"',
    'C-Clef.L4': '\\revert Staff.Clef.stencil \\clef "tenor"',
    'C-Clef.L5': '\\revert Staff.Clef.stencil \\clef "baritone"',
}

lilypond_rhythm_mapping = {
    'blank': '',
    'epsilon': '',
    'barline': '\\bar "|"',
    # 'C-Clef': '\\clef "alto"', # caso particolare gestito a parte
    'startSlur': '(',
    'endSlur': ')',
    'timeSig_common': '\\revert Staff.TimeSignature.stencil \\time 4/4',

    '32thRest': 'r32',
    '16thRest': 'r16',
    'eighthRest': 'r8',
    'quarterRest': 'r4',
    'halfRest': 'r2',
    'wholeRest': 'r1',
    'doubleWholeRest': 'r1*2',
    'quadrupleWholeRest': 'r1*4',

    '32thNote': '{note}{octave}32',
    '16thNote': '{note}{octave}16',
    'eighthNote': '{note}{octave}8',
    'quarterNote': '{note}{octave}4',
    'halfNote': '{note}{octave}2',
    'wholeNote': '{note}{octave}1',

    'dotted32thNote': '{note}{octave}32.',
    'dotted16thNote': '{note}{octave}16.',
    'dottedEighthNote': '{note}{octave}8.',
    'dottedQuarterNote': '{note}{octave}4.',
    'dottedHalfNote': '{note}{octave}2.',
    'dottedWholeNote': '{note}{octave}1.',

    'flat32thNote': '{note}es{octave}32',
    'flat16thNote': '{note}es{octave}16',
    'flatEighthNote': '{note}es{octave}8',
    'flatQuarterNote': '{note}es{octave}4',
    'flatHalfNote': '{note}es{octave}2',
    'flatWholeNote': '{note}es{octave}1',

    'sharp32thNote': '{note}is{octave}32',
    'sharp16thNote': '{note}is{octave}16',
    'sharpEighthNote': '{note}is{octave}8',
    'sharpQuarterNote': '{note}is{octave}4',
    'sharpHalfNote': '{note}is{octave}2',
    'sharpWholeNote': '{note}is{octave}1',

    'natural32thNote': '{note}{octave}!32',
    'natural16thNote': '{note}{octave}!16',
    'naturalEighthNote': '{note}{octave}!8',
    'naturalQuarterNote': '{note}{octave}!4',
    'naturalHalfNote': '{note}{octave}!2',
    'naturalWholeNote': '{note}{octave}!1',

    'dottedFlat32thNote': '{note}es{octave}32.',
    'dottedFlat16thNote': '{note}es{octave}16.',
    'dottedFlatEighthNote': '{note}es{octave}8.',
    'dottedFlatQuarterNote': '{note}es{octave}4.',
    'dottedFlatHalfNote': '{note}es{octave}2.',
    'dottedFlatWholeNote': '{note}es{octave}1.',

    'dottedSharp32thNote': '{note}is{octave}32.',
    'dottedSharp16thNote': '{note}is{octave}16.',
    'dottedSharpEighthNote': '{note}is{octave}8.',
    'dottedSharpQuarterNote': '{note}is{octave}4.',
    'dottedSharpHalfNote': '{note}is{octave}2.',
    'dottedSharpWholeNote': '{note}is{octave}1.',

    'dottedNatural32thNote': '{note}{octave}!32.',
    'dottedNatural16thNote': '{note}{octave}!16.',
    'dottedNaturalEighthNote': '{note}{octave}!8.',
    'dottedNaturalQuarterNote': '{note}{octave}!4.',
    'dottedNaturalHalfNote': '{note}{octave}!2.',
    'dottedNaturalWholeNote': '{note}{octave}!1.'
}

ordered_positions = ['L0', 'S0', 'L1', 'S1', 'L2', 'S2', 'L3', 'S3', 'L4', 'S4', 'L5', 'S5', 'L6', 'S6']
notes = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
octaves = [',,,', ',,', ',', '', '\'', '\'\'', '\'\'\'']
clef_position_offsets = {
    'treble': 0,  # chiave di violino
    'C-Clef.L1': -2,  # chiave di soprano
    'C-Clef.L2': -4,  # chiave di mezzo-soprano
    'C-Clef.L3': -6,  # chiave di alto
    'C-Clef.L4': -8,  # chiave di tenore
    'C-Clef.L5': -10  # chiave di baritono
}
treble_base_octave_index = 4
c_treble_base_index = treble_base_octave_index * len(ordered_positions)


# Funzione che a partire dalla posizione della nota sul pentagramma e la chiave in uso
# restituisce la nota musicale e la sua ottava
def get_note_octave(position, clef="treble"):
    if position in ['blank', 'epsilon', 'noNote']:
        return ['', '']

    absolute_position_index = c_treble_base_index + ordered_positions.index(position)
    adjusted_absolute_position_index = absolute_position_index + clef_position_offsets[clef]

    note_index = adjusted_absolute_position_index % len(notes)
    octave_index = 4 + (adjusted_absolute_position_index - c_treble_base_index) // len(notes)

    note = notes[note_index]
    octave = octaves[octave_index]

    return [note, octave]


# Funzione che a partire dal simbolo in sintassi per OMR e chiave in uso restituisce la rispettiva sintassi Lilypond
def omr_to_lilypond(symbol, clef):
    if symbol == "epsilon":
        return

    if 'C-Clef' in symbol:
        return lilypond_cclef_mapping.get(symbol)

    rhythm, pitch = symbol.split('.')
    duration = lilypond_rhythm_mapping.get(rhythm, '')
    note = get_note_octave(position=pitch, clef=clef)

    return duration.replace("{note}", note[0]).replace("{octave}", note[1])


# Funzione che converte un pdf in immagine e la ritaglia al contenuto aggiungendo un certo margine
def pdf_to_cropped_png(pdf_path, output_path, margin=10):
    # Converti il PDF in immagine
    page = convert_from_path(pdf_path)[0]

    # Converti l'immagine PIL in array numpy per OpenCV
    img_np = np.array(page)

    # Converti in scala di grigi
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Trova i contorni
    _, thresh = cv2.threshold(gray, 240, 255,
                              cv2.THRESH_BINARY_INV)  # 240 è una soglia approssimativa, potrebbe aver bisogno di regolazione
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcola il bounding box basato sui contorni
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))

    # Ritaglia l'immagine usando PIL
    cropped = page.crop((x, y, x + w, y + h))

    # Aggiunge un margine
    cropped_with_margin = ImageOps.expand(cropped, border=margin, fill='white')

    # Salva l'immagine PNG
    cropped_with_margin.save(f"{output_path}.png", "PNG")


excluding_values = ['blank', 'epsilon', 'barline', 'C-Clef', 'startSlur', 'endSlur', 'timeSig_common']
symbols = [value for value in human_readable_rhythm_mapping.values() if value not in excluding_values]
notesOfLength = {
    "32thNote": [value for value in human_readable_rhythm_mapping.values() if "32thNote" in value],
    "16thNote": [value for value in human_readable_rhythm_mapping.values() if "16thNote" in value],
    "eighthNote": [value for value in human_readable_rhythm_mapping.values() if "eighthNote".lower() in value.lower()]
}
pitches = list(human_readable_pitch_mapping.values())[3:]  # Exclude blank, epsilon, noNote


def generate_dataset(annotations_output, num_bars=1, starting_from=0, num_symbols=random.randint(3, 8)):
    thresh_out = open(annotations_output, 'w')
    for idx in tqdm(range(starting_from, starting_from + num_bars)):
        annotation = []
        thresh_out.write(f"00_{idx}$166|")
        lilypond_output = f"words/00_{idx}.ly"
        lily_out = open(lilypond_output, 'w')
        lily_out.write(header)
        clef = 'treble'

        # Inserisce chiave e indicazione tempo, o linea di battuta
        if random.randint(1, 2) == 1:
            clef = "C-Clef." + random.choice(['L1', 'L2', 'L3', 'L4', 'L5'])
            thresh_out.write(clef)
            thresh_out.write("~epsilon~timeSig_common.noNote~")
            lily_out.write(
                "\\cadenzaOn\n" +
                lilypond_cclef_mapping.get(clef) + " \\revert Staff.TimeSignature.stencil \\time 4/4\n"
            )
        else:
            thresh_out.write("barline.noNote~")
            lily_out.write(
                "\\once \n" +
                "\\override Score.TimeSignature.transparent = ##t \n" +
                "\\time 1/4 \n" +
                "s4 \n" +
                "\\cadenzaOn\n "
            )

        # Genera casualmente un elenco di simboli ammissibili
        for _ in range(num_symbols):
            rhythm = random.choice(symbols)
            pitch = random.choice(pitches) if "Note" in rhythm else "noNote"
            annotation.append(f"{rhythm}.{pitch}")

        # Se la sequenza è abbastanza lunga, inserisce casualmente una legatura
        if len(annotation) > 6 and random.choice([True, False]):
            start_slur_index = random.randint(1, 4)
            end_slur_index = random.randint(start_slur_index + 2, len(annotation) - 1)
            annotation.insert(start_slur_index, "startSlur.noNote")
            annotation.insert(end_slur_index, "endSlur.noNote")

        # Inserisce linea di battuta finale
        annotation.append("barline.noNote")

        # Traduce i simboli estratti casualmente in sintassi Lilypond e li scrive sul file .ly
        for symbol in annotation:
            lilypond_syntax = omr_to_lilypond(symbol, clef)
            lily_out.write(lilypond_syntax + ' ')

        # Chiude il file .ly inserendo un segmento di pentagramma aggiuntivo
        lily_out.write(
            "\n" +
            "\\cadenzaOff \n"
            "\\once \n" +
            "\\override Score.TimeSignature.transparent = ##t \n" +
            "\\time 1/16 \n" +
            "s32 \n"
            "}"
        )
        lily_out.close()

        thresh_out.write("~epsilon~".join(annotation) + '\n')

        # Conversione file .ly in .pdf
        result = subprocess.run(["lilypond", "-o", f"words/00_{idx}", lilypond_output], capture_output=True, text=True)

        pdf_to_cropped_png(f"words/00_{idx}.pdf", f"words/00_{idx}")

        os.remove(f"words/00_{idx}.pdf")
        os.remove(f"words/00_{idx}.ly")


times_historical_dataset = 5
generate_dataset("lilypond.train.thresh", num_bars=147 * times_historical_dataset, starting_from=0)
generate_dataset("lilypond.valid.thresh", num_bars=49 * times_historical_dataset,
                 starting_from=147 * times_historical_dataset)
generate_dataset("lilypond.test.thresh", num_bars=49 * times_historical_dataset,
                 starting_from=(147 + 49) * times_historical_dataset)
