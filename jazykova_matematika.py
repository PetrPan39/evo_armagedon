import numpy as np

# Slovní druhy: 1=zájmeno, 2=podstatné jméno, 3=sloveso, 4=tázací, 5=příslovce, 6=přídavné jméno, 7=rozkaz, 8=citoslovce
# Vlastnosti: [druh, pád, číslo, osoba, čas, význam, emoce, funkce]
slovnik = {
    "já":      [1, 1, 1, 1, 0, 0, 0, 0],
    "ty":      [1, 1, 1, 2, 0, 0, 0, 0],
    "pes":     [2, 1, 1, 3, 0, 1, 0, 0],
    "domů":    [2, 4, 1, 0, 0, 2, 0, 0],
    "jde":     [3, 1, 1, 3, 1, 1, 0, 1],
    "kam":     [4, 1, 1, 0, 0, 2, 0, 2],
    "kdo":     [4, 1, 1, 0, 0, 0, 0, 2],
    "půjdeme": [3, 1, 2, 1, 2, 1, 0, 1],
    "dnes":    [5, 0, 0, 0, 0, 3, 0, 0],
    "venku":   [5, 0, 0, 0, 0, 2, 0, 0],
    "pekně":   [6, 0, 0, 0, 0, 4, 1, 0],
    "dej":     [7, 1, 2, 2, 1, 5, 0, 3],
    "Haf":     [8, 0, 0, 0, 0, 1, 2, 4],
}

# Vztahy/vazby (vztah, síla, význam/funkce)
vazby = {
    ("kam", "půjdeme"): ("tázací", 0.9, "dotaz na směr"),
    ("pes", "jde"): ("akce", 0.8, "pes vykonal pohyb"),
    ("dnes", "je"): ("čas", 1.0, "časový údaj"),
    ("venku", "pekně"): ("popis", 1.0, "stav počasí"),
    ("půjdeme", "venku"): ("akce", 0.7, "směřování ven"),
}

# Věta = seznam slov (pro demo)
vety = [
    ["pes", "jde", "domů"],
    ["kam", "půjdeme"],
    ["dnes", "je", "pekně"],
    ["dej", "Haf"],
    ["já", "půjdeme", "domů"],
]

# Funkce: získá matici slovních vlastností
def matice_vety(veta):
    return np.array([slovnik[w] for w in veta if w in slovnik])

# Typ věty: jednoduchá logika podle vzoru
def typ_vety(veta):
    vzor = [slovnik[w][0] for w in veta if w in slovnik]
    if vzor and vzor[0] == 4:
        return "tázací"
    elif 7 in vzor:
        return "rozkazovací"
    elif vzor and vzor[0] == 8:
        return "citoslovce"
    elif 3 in vzor and vzor[0] in [1,2]:
        return "oznamovací"
    return "jiný/neurčitý"

# Hledej nejlogičtější vazby ve větě
def hledej_vazby(veta):
    vztahy = []
    for i, slovo_a in enumerate(veta[:-1]):
        for j, slovo_b in enumerate(veta[i+1:], i+1):
            key = (slovo_a, slovo_b)
            if key in vazby:
                vztahy.append((slovo_a, slovo_b) + vazby[key])
    return vztahy

# Analýza a tisk
for veta in vety:
    print("Věta:", " ".join(veta))
    print("Typ věty:", typ_vety(veta))
    print("Matice vlastností:\n", matice_vety(veta))
    vztahy = hledej_vazby(veta)
    if vztahy:
        print("Vazby:", vztahy)
    print("-" * 40)
