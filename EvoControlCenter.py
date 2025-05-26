from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QCheckBox, QPushButton, QMessageBox
import sys, json, os

CONFIG_FILE = "evo_config.json"

class EvoControlCenter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EVO Control Center")
        self.setGeometry(300, 100, 450, 500)
        self.layout = QVBoxLayout()
        self.modules = {
            "Memory Engine": True,
            "Emotion Mapping": True,
            "Decision Tree": False,
            "Motor Output": False,
            "Procedural Generator": False,
            "Telemetry Collector": False,
            "Text Parser (NLP)": True,
            "Emotion Vectors": True,
            "Interaction DB": False,
            "Scene Context Map": False,
            "Memory Tracker": False
        }
        self.checks = {}
        self.load_config()
        self.build_ui()

    def build_ui(self):
        self.layout.addWidget(QLabel("Select active EVO modules:"))
        for mod in self.modules:
            cb = QCheckBox(mod)
            cb.setChecked(self.modules[mod])
            self.layout.addWidget(cb)
            self.checks[mod] = cb

        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save_config)
        self.layout.addWidget(save_btn)

        self.setLayout(self.layout)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                self.modules.update(json.load(f))

    def save_config(self):
        config = {mod: self.checks[mod].isChecked() for mod in self.modules}
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        QMessageBox.information(self, "Saved", "Configuration saved to evo_config.json")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = EvoControlCenter()
    win.show()
    sys.exit(app.exec_())
