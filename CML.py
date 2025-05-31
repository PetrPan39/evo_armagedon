from CML_A import CML as CML_A
from CML_B import CML_B
from CML_C import CML_C

class CML:
    def __init__(self):
        self.plugin_manager = CML_B("plugins")
        self.premier = CML_A(n=10)  # BEZE ZMĚNY!
        self.speaker = CML_C(self, self.plugin_manager)

    def predikuj_a_optim(self, text, vektor, target=None):
        # Prezident může volat evoluci nebo jen predikci – záleží na zadání
        if target is not None:
            output = self.premier.learn(vektor, target)
        else:
            output = self.premier.predict(vektor)
        return output

    def testuj_evoluci(self, sentences, n_iter=10):
        # Test: proveď n_iter evolučních kroků na datech z CML_C
        history = []
        for i, text in enumerate(sentences):
            vektor = self.speaker.vektorizuj(text)
            # Simulujeme target (např. suma vektoru, nebo nějaký label)
            target = [sum(vektor)] * len(vektor)
            output = self.premier.learn(vektor, target)
            history.append({
                "step": i,
                "input": text,
                "vector": vektor,
                "target": target,
                "output": output.tolist() if hasattr(output, "tolist") else output
            })
        return history

    def vyhodnoceni(self):
        # Vyhodnotí úspěšnost optimalizace/evoluce premiéra
        best_score = getattr(self.premier, "_best_score", None)
        steps = getattr(self.premier, "_evolution_level", None)
        return {"best_score": best_score, "evolution_steps": steps}