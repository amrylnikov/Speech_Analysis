import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLineEdit, QLabel
import argparse

class Window2(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('second window')
        self.setFixedWidth(500)
        self.setStyleSheet("""
            QLineEdit{
                font-size: 30px
            }
            QPushButton{
                font-size: 30px
            }
            """)
        mainLayout = QVBoxLayout()

        self.input1 = QLineEdit()
        self.input2 = QLineEdit()
        self.input3 = QLineEdit()
        self.input4 = QLineEdit()
        self.input5 = QLineEdit()
        self.input6 = QLineEdit()
        mainLayout.addWidget(self.input1)
        mainLayout.addWidget(self.input2)
        mainLayout.addWidget(self.input3)
        mainLayout.addWidget(self.input4)
        mainLayout.addWidget(self.input5)
        mainLayout.addWidget(self.input6)

        self.closeButton = QPushButton('Close')
        self.closeButton.clicked.connect(self.close)
        mainLayout.addWidget(self.closeButton)

        self.setLayout(mainLayout)

    def displayInfo(self):
        self.show()
