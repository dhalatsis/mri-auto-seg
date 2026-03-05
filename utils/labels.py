"""Muscle label definitions shared across the project."""

# Label ID -> muscle name (used in segmentation outputs)
LABELS = {
    2: "ANC", 3: "APL", 4: "ECRB", 5: "ECRL", 6: "ECU",
    7: "ED", 8: "EDM", 9: "EPL", 10: "FCR", 11: "FCU",
    12: "FDP", 14: "FDS", 15: "FPL", 16: "PL", 17: "PQ",
    18: "PT", 19: "SUP",
}

# Muscle name -> label ID (used in data conversion / ROI parsing)
MUSCLE_LABELS = {
    "ANC": 2,   # Anconeus
    "APL": 3,   # Abductor Pollicis Longus
    "ECRB": 4,  # Extensor Carpi Radialis Brevis
    "ECRL": 5,  # Extensor Carpi Radialis Longus
    "ECU": 6,   # Extensor Carpi Ulnaris
    "ED": 7,    # Extensor Digitorum
    "EDM": 8,   # Extensor Digiti Minimi
    "EPL": 9,   # Extensor Pollicis Longus
    "FCR": 10,  # Flexor Carpi Radialis
    "FCU": 11,  # Flexor Carpi Ulnaris
    "FDP": 12,  # Flexor Digitorum Profundus
    "FDS": 14,  # Flexor Digitorum Superficialis
    "FPL": 15,  # Flexor Pollicis Longus
    "PL": 16,   # Palmaris Longus
    "PQ": 17,   # Pronator Quadratus
    "PT": 18,   # Pronator Teres
    "SUP": 19,  # Supinator
}

# Reverse mapping: name -> ID
NAME_TO_LID = {v: k for k, v in LABELS.items()}
