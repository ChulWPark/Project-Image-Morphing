# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MorphingGUI.ui'
#
# Created: Tue Apr 17 19:05:55 2018
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(873, 835)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnLoadS = QtGui.QPushButton(self.centralwidget)
        self.btnLoadS.setGeometry(QtCore.QRect(0, 10, 161, 27))
        self.btnLoadS.setObjectName("btnLoadS")
        self.btnLoadE = QtGui.QPushButton(self.centralwidget)
        self.btnLoadE.setGeometry(QtCore.QRect(470, 10, 151, 27))
        self.btnLoadE.setObjectName("btnLoadE")
        self.horizontalSlider = QtGui.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(50, 380, 741, 20))
        font = QtGui.QFont()
        font.setWeight(50)
        font.setItalic(False)
        font.setBold(False)
        self.horizontalSlider.setFont(font)
        self.horizontalSlider.setAutoFillBackground(False)
        self.horizontalSlider.setMaximum(20)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(630, 350, 101, 17))
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(150, 350, 111, 17))
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(770, 400, 21, 17))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 380, 41, 17))
        self.label_4.setObjectName("label_4")
        self.graphicsView_1 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_1.setGeometry(QtCore.QRect(0, 40, 400, 300))
        self.graphicsView_1.setObjectName("graphicsView_1")
        self.checkBox = QtGui.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(370, 350, 121, 22))
        self.checkBox.setObjectName("checkBox")
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(60, 400, 21, 17))
        self.label_6.setObjectName("label_6")
        self.graphicsView_2 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(470, 40, 400, 300))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_3 = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView_3.setGeometry(QtCore.QRect(240, 420, 400, 300))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.label_7 = QtGui.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(380, 730, 121, 17))
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.btnBlend = QtGui.QPushButton(self.centralwidget)
        self.btnBlend.setGeometry(QtCore.QRect(380, 760, 121, 21))
        self.btnBlend.setObjectName("btnBlend")
        self.lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(800, 380, 71, 21))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 873, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.btnLoadS.setText(QtGui.QApplication.translate("MainWindow", "Load Starting Image ...", None, QtGui.QApplication.UnicodeUTF8))
        self.btnLoadE.setText(QtGui.QApplication.translate("MainWindow", "Load Ending Image ...", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("MainWindow", "Ending Image", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("MainWindow", "Starting Image", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("MainWindow", "1.0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("MainWindow", "Alpha", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox.setText(QtGui.QApplication.translate("MainWindow", "Show Triangles", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("MainWindow", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("MainWindow", "Blending Result", None, QtGui.QApplication.UnicodeUTF8))
        self.btnBlend.setText(QtGui.QApplication.translate("MainWindow", "Blend", None, QtGui.QApplication.UnicodeUTF8))
