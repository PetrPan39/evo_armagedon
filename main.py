from CML import CML

# Testovací data
sentences = [
    "Jsem nadšený z našeho pokroku!",
    "Cítím se klidně a spokojeně.",
    "To mě opravdu potěšilo, děkuji.",
    "Mám špatný pocit z tohoto rozhovoru.",
    "Tohle není fér, jsem naštvaný.",
    "Doufám, že to dobře dopadne.",
    "Nerozumím ti, můžeš to vysvětlit znovu?",
    "Tohle je fantastické zjištění!",
    "Připadám si ztraceně a nejistě.",
    "Skvělé! Právě jsme dokončili simulaci."
]

mozek = CML()
evoluce = mozek.testuj_evoluci(sentences)
vyhodnoceni = mozek.vyhodnoceni()

print("=== Evoluční kroky ===")
for step in evoluce:
    print(step)

print("\n=== Vyhodnocení optimalizace ===")
print(vyhodnoceni)