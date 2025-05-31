
# === MODULE: core/CML_B.py ===
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.CML_A import CML as CMLCore
from core.CML_C import CML as LexicalCore


class CMLEvolutionVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_cml_a_plane(self, evolution_steps, matrix_fitness):
        xs = evolution_steps
        ys = np.full_like(xs, fill_value=0)
        zs = matrix_fitness
        self.ax.plot(xs, ys, zs, label='CML_A Evolution', color='blue')
        self.ax.set_xlabel('Krok')
        self.ax.set_ylabel('Placeholder')
        self.ax.set_zlabel('Fitness')
        self.ax.legend()
        plt.show()


class CMLBridge:
    def __init__(self, vector_dim=8):
        self.core = CMLCore(n=vector_dim)
        self.lex = LexicalCore()

    def process_text(self, text, target=None):
        vector = self.lex._vectorize(text)
        if target:
            target_vec = self.lex._vectorize(target)
            self.core._adaptive_optimize(vector, target_vec)
        else:
            prediction = self.core._nonlinear_predict(vector)
            return prediction

    def evaluate_emotion(self, text):
        # Placeholder – může volat Evo_Emocion_Bank nebo interní logiku
        return self.lex._vectorize(text)[4]  # index 4 = "emoce"

    def encode_text(self, text):
        return self.lex._vectorize(text)

    def visualize(self):
        viz = CMLEvolutionVisualizer()
        steps = list(range(len(self.core._fitness_history)))
        fitness = self.core._fitness_history
        viz.plot_cml_a_plane(np.array(steps), np.array(fitness))
