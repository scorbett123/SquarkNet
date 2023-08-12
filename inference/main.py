import typing
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QGridLayout, QHBoxLayout, QLabel, QFileDialog, QVBoxLayout, QFrame, QSlider
from PyQt6 import QtCore, QtWidgets
import sys
from model import models
import inference


class FileSelectionWidget(QFrame):
    file_chosen = QtCore.pyqtSignal(str)
    def __init__(self, name: str, filter: str, open=True, force_ends="") -> None:
        super().__init__()
        self.filter = filter
        self.name = name
        self.open = open
        self.force_ends = force_ends
        
        self.setObjectName("FileSelectionWidget")

        self.setStyleSheet("""
                           FileSelectionWidget {
                                    background-color: #b9b9b9; 
                                    color: white; 
                                    font-size: 20px;
                                    margin: 2px;
                                    border-radius: 5px;
                           }""")

        layout = QHBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        self.select_button = QPushButton("Select")
        self.select_button.clicked.connect(self.choose_file)
        layout.addWidget(self.select_button)
        
        self._file_name = QLabel("No file selected")
        layout.addWidget(self._file_name)

        self.main_widget = QWidget()
        self.setLayout(layout)

        self.setMaximumHeight(50)
        self.filename = ""

    def choose_file(self):
        if self.open:
            fname = QFileDialog.getOpenFileName(self, 'Open file', 
            '', self.filter)[0]
            if not fname.endswith(self.force_ends):
                fname += self.force_ends
        else:
            fname = QFileDialog.getSaveFileName(self, 'Save file', 
            '', self.filter)[0]
        self.filename = fname
        self._file_name.setText(fname if fname else "No file selected")
        self.file_chosen.emit(fname)


class ModelSelectionWidget(QWidget):
    updated = QtCore.pyqtSignal(models.Models)
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Model Selection"))
        self.file_select = FileSelectionWidget("Model File", "Model files ()")
        self.file_select.file_chosen.connect(self.model_update)

        layout.addWidget(self.file_select)
        self.setLayout(layout)

    def model_update(self):
        m = models.Models.load(self.file_select.filename)
        self.updated.emit(m)

class ModelStatsWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.epochs = QLabel("Epochs: ?")
        layout.addWidget(self.epochs)
        layout.addWidget(QLabel("Model Stats"))
        self.setLayout(layout)

    def update(self, models: models.Models):
        self.epochs.setText(f"Epochs: {models.epochs}")


class EncodeDecodeWidget(QWidget):
    def __init__(self, name, input_filter, output_filter, output_type):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(name))
        self.input_select = FileSelectionWidget("Input", input_filter)
        self.output_select = FileSelectionWidget("Output", output_filter, open=False, force_ends=output_type)
        self.run_button = QPushButton("Run")
        self.run_button.setDisabled(True)
        self.run_button.clicked.connect(self.run)
        layout.addWidget(self.input_select)
        layout.addWidget(self.output_select)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

        self.model = None

    def model_set(self, model):
        self.model = model
        if model != None:
            self.run_button.setDisabled(False)

    def run(self):
        pass


class EncodeWidget(EncodeDecodeWidget):
    def __init__(self):
        super().__init__("Encode", "Audio files (*.wav)", "Audio files (*.sc)", output_type=".sc")

    def run(self):
        inference.wav_to_sc(self.input_select.filename, self.output_select.filename, self.model)

class DecodeWidget(EncodeDecodeWidget):
    def __init__(self):
        super().__init__("Decode", "Audio files (*.sc)", "Audio files (*.wav)", output_type=".wav")

    def run(self):
        inference.sc_to_wav(self.input_select.filename, self.output_select.filename, self.model)


class EncodeDecodeContainer(QFrame):
    def __init__(self):
        super().__init__()
        
        self.switch = QPushButton("Mode")
        self.switch.setCheckable(True)
        self.switch.clicked.connect(self.switch_ec)

        self.encode = EncodeWidget()
        self.decode = DecodeWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.switch)
        layout.addWidget(self.encode)
        layout.addWidget(self.decode)

        self.decode.setVisible(False)

        self.setLayout(layout)

    
    def switch_ec(self, value):
        self.decode.setVisible(value)
        self.encode.setVisible(not value)

    
    def model_set(self, model):
        self.encode.model_set(model)
        self.decode.model_set(model)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        self.model_select = ModelSelectionWidget()
        self.model_stats = ModelStatsWidget()
        self.encode_decode = EncodeDecodeContainer()
        
        layout = QGridLayout()
        layout.addWidget(self.model_select, 0, 0, 2, 2)
        layout.addWidget(self.model_stats, 2, 0, 2, 2)
        layout.addWidget(self.encode_decode, 0, 2, 4, 2)

        self.model_select.updated.connect(self.model_stats.update)
        self.model_select.updated.connect(self.on_model_select)
        self.model_select.updated.connect(self.encode_decode.model_set)

        self.model = None

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def on_model_select(self, model):
        self.model = model



app = QApplication(sys.argv)

window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()