import typing
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QGridLayout, QHBoxLayout, QLabel, QFileDialog, QVBoxLayout, QFrame, QSlider
from PyQt6.QtCore import QSize, Qt
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPalette, QColor, QFont
import sys
import PyQt6


class FileSelectionWidget(QFrame):
    file_chosen = QtCore.pyqtSignal(str)
    def __init__(self, name: str, filter: str, open=True) -> None:
        super().__init__()
        self.filter = filter
        self.name = name
        self.open = open
        
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
            '', self.filter)
        else:
            fname = QFileDialog.getSaveFileName(self, 'Save file', 
            '', self.filter)
        self.filename = fname[0]
        self._file_name.setText(fname[0] if fname[0] else "No file selected")
        self.file_chosen.emit(fname[0])


class ModelSelectionWidget(QWidget):
    updated = QtCore.pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Model Selection"))
        self.file_select = FileSelectionWidget("Model File", "Model files ()")
        self.file_select.file_chosen.connect(self.model_update)

        layout.addWidget(self.file_select)
        self.setLayout(layout)

    def model_update(self):
        self.updated.emit({"filename": self.file_select.filename})

class ModelStatsWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Model Stats"))
        self.setLayout(layout)

class EncodeWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Encode"))
        self.input_select = FileSelectionWidget("Input", "Audio files (*.wav)")
        self.output_select = FileSelectionWidget("Output", "Audio files (*.sc)", open=False)
        layout.addWidget(self.input_select)
        layout.addWidget(self.output_select)
        self.setLayout(layout)

class DecodeWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Decode"))
        self.input_select = FileSelectionWidget("Input", "Audio files (*.sc)")
        self.output_select = FileSelectionWidget("Output", "Audio files (*.wav)", open=False)
        layout.addWidget(self.input_select)
        layout.addWidget(self.output_select)
        self.setLayout(layout)


class EncodeDecodeWidget(QFrame):
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        self.model_select = ModelSelectionWidget()
        self.model_stats = ModelStatsWidget()
        self.encode_decode = EncodeDecodeWidget()
        
        layout = QGridLayout()
        layout.addWidget(self.model_select, 0, 0, 2, 2)
        layout.addWidget(self.model_stats, 2, 0, 2, 2)
        layout.addWidget(self.encode_decode, 0, 2, 4, 2)

        self.model_select.updated.connect(self.on_model_select)


        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def on_model_select(self, model_params):
        print(model_params)



app = QApplication(sys.argv)

window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()