from langdetect import detect

class CML_C:
    def __init__(self, president, plugin_manager):
        self.president = president        # reference na prezidenta (CML)
        self.plugin_manager = plugin_manager  # správce pluginů (CML_B)

    def vektorizuj(self, text):
        # Zde lze volat pluginy na vektorizaci, morfologii apod.
        # Ukázka: použij plugin na statistický průměr (demo)
        words = text.split()
        word_lengths = [len(w) for w in words]
        if self.plugin_manager.get_plugin("statistika", "prumer"):
            avg_len = self.plugin_manager.call("statistika", "prumer", word_lengths)
        else:
            avg_len = sum(word_lengths)/len(word_lengths)
        return [len(words), avg_len]    # Demo vektor: počet slov, průměrná délka

    def analyzuj(self, text):
        lang = detect(text)
        vektor = self.vektorizuj(text)
        # Prezidentovi předáme (vektor, jazyk) – on rozhodne, co dál (učení/evoluce/test)
        result = self.president.predikuj_a_optim(text, vektor)
        return {
            "lang": lang,
            "vector": vektor,
            "president_result": result
        }