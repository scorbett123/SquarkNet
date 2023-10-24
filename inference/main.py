import typing
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QGridLayout, QHBoxLayout, QLabel, QFileDialog, QVBoxLayout, QFrame, QSlider, QProgressBar, QMessageBox
from PyQt6 import QtCore, QtWidgets
import sys
from model import models
import inference
import math
import time
from typing import Callable
from threading import Thread
import traceback

class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(Exception)
    progress = QtCore.pyqtSignal(float)
    def __init__(self, callable, *args, **kwargs) -> None:
        super().__init__()
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.kwargs["progress_callback"] = self.progress.emit
        try:
            self.callable(*self.args, **self.kwargs)
        except Exception as e:
            traceback.print_exception(e)
            self.error.emit(e)
        self.finished.emit()


class ProgressBar(QWidget):

    def __init__(self, callable, args, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self.pbar = QProgressBar()
        layout = QVBoxLayout()

        layout.addWidget(self.pbar)
        self.setLayout(layout)

        self._thread = QtCore.QThread()
        self.worker = Worker(callable, *args)

        self.worker.moveToThread(self._thread)

        self._thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update)
        self.worker.finished.connect(self.finished)
        self.worker.error.connect(self.exception)

        #  house keeping
        self.worker.finished.connect(self._thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()
        self.show()

        
        self.main_window.setDisabled(True)


    def update(self, amount):
        self.pbar.setValue(int(100 * amount))

    def finished(self):
        self.main_window.setDisabled(False)
        self.close()

    def exception(self, e):
        if e is inference.InvalidHashException:
            button = QMessageBox.critical(
                None,
                "Error",
                "This file was encoded with a different model, please select the correct model",
                buttons=QMessageBox.StandardButton.Ok
            )
        else:
            button = QMessageBox.critical(
                None,
                "Error",
                "An error has occured, try again",
                buttons=QMessageBox.StandardButton.Ok
            )



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
        else:
            fname = QFileDialog.getSaveFileName(self, 'Save file', 
            '', self.filter)[0]
        
        if (not fname.endswith(self.force_ends)) and fname != "":
            fname += self.force_ends
        self.filename = fname
        self._file_name.setText(fname if fname else "No file selected")
        self.file_chosen.emit(fname)


class ModelSelectionWidget(QWidget):
    updated = QtCore.pyqtSignal(models.Models)
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Model Selection"))
        self.file_select = FileSelectionWidget("Model File", "Model files ()", force_ends=".saved")
        self.file_select.file_chosen.connect(self.model_update)

        layout.addWidget(self.file_select)
        self.setLayout(layout)

    def model_update(self):
        m = models.Models.load(self.file_select.filename)
        self.updated.emit(m)

class ModelStatsWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Model Stats"))
        self.epochs = QLabel("Epochs: ?")
        self.ncodes = QLabel("Codes: ?")
        self.nbooks = QLabel("Codebooks: ?")
        self.bitrate = QLabel("Bitrate: ?")

        layout.addWidget(self.epochs)
        layout.addWidget(self.ncodes)
        layout.addWidget(self.nbooks)
        layout.addWidget(self.bitrate)
        self.setLayout(layout)

    def update(self, models: models.Models):
        self.epochs.setText(f"Epochs: {models.epochs}")
        self.ncodes.setText(f"Codebooks: {models.ncodes}")
        self.nbooks.setText(f"Books: {models.nbooks}")

        bitrate = 16000 / models.ctx_len * models.nbooks * math.log2(models.ncodes) / 1000
        self.bitrate.setText(f"Bitrate: {bitrate:.2f} kbps")

        # bitrate = math.ceil(math.log2(models.ncodes) * models.nbooks)
        # self.bitrate.setText(f"Bitrate: {round(bitrate / 1000, 1)} kbps")


class EncodeDecodeWidget(QWidget):
    def __init__(self, name, input_filter, output_filter, output_type, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout()
        layout.addWidget(QLabel(name))
        self.input_select = FileSelectionWidget("Input", input_filter)
        self.output_select = FileSelectionWidget("Output", output_filter, open=False, force_ends=output_type)

        self.input_select.file_chosen.connect(self.update_button)
        self.output_select.file_chosen.connect(self.update_button)

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
        self.update_button()

    def update_button(self):
        if self.model != None and self.input_select.filename != "" and self.output_select.filename != "":
            self.run_button.setDisabled(False)
        else:
            self.run_button.setDisabled(True)

    def run(self):
        pass


class EncodeWidget(EncodeDecodeWidget):
    def __init__(self, main_window):
        super().__init__("Encode", "Audio files (*.wav)", "Audio files (*.sc)", ".sc", main_window)

    def run(self):
        self.progress = ProgressBar(inference.wav_to_sc,(self.input_select.filename, self.output_select.filename, self.model), self.main_window)
        self.progress.show()
        # progress has can't be a local variable as otherwise gc becomes greedy and "eats" it before it's been displayed


class DecodeWidget(EncodeDecodeWidget):
    def __init__(self, main_window):
        super().__init__("Decode", "Audio files (*.sc)", "Audio files (*.wav)", ".wav", main_window)

    def run(self):
        self.progress = ProgressBar(inference.sc_to_wav, (self.input_select.filename, self.output_select.filename, self.model), self.main_window)
        self.progress.show()


class EncodeDecodeContainer(QFrame):
    def __init__(self, main_window):
        super().__init__()
        
        self.switch = QPushButton("Mode")
        self.switch.setCheckable(True)
        self.switch.clicked.connect(self.switch_ec)

        self.encode = EncodeWidget(main_window)
        self.decode = DecodeWidget(main_window)

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
        self.encode_decode = EncodeDecodeContainer(main_window=self)
        
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
        self.setFixedSize(600, 300)

    def on_model_select(self, model):
        self.model = model


app = QApplication(sys.argv)

window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()