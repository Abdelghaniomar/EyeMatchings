# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface2.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(" Dialog")
        Dialog.resize(800, 600)
        Dialog.setStyleSheet("background-color:rgb(85, 0, 255)")
        self.centralwidget = QtWidgets.QWidget( Dialog)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(310, 30, 91, 23))
        self.pushButton.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(410, 30, 221, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 90, 111, 41))
        self.pushButton_2.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 140, 111, 41))
        self.pushButton_3.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(50, 190, 111, 41))
        self.pushButton_6.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(50, 240, 111, 41))
        self.pushButton_8.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(50, 290, 111, 41))
        self.pushButton_9.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(50, 340, 111, 41))
        self.pushButton_10.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton_10.setObjectName("pushButton_10")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(390, 70, 256, 192))
        self.graphicsView.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.graphicsView.setObjectName("graphicsView")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(400, 280, 111, 41))
        self.pushButton_11.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.pushButton_11.setObjectName("pushButton_11")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(410, 330, 256, 192))
        self.graphicsView_2.setStyleSheet("background-color :rgb(255, 255, 255)")
        self.graphicsView_2.setObjectName("graphicsView_2")
       # Dialog.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar( Dialog)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
       # Dialog.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar( Dialog)
        self.statusbar.setObjectName("statusbar")
        #Dialog.setStatusBar(self.statusbar)

        self.retranslateUi( Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self,  Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Load"))
        self.pushButton_2.setText(_translate("MainWindow", "Histogramme"))
        self.pushButton_3.setText(_translate("MainWindow", "Lissage"))
        self.pushButton_6.setText(_translate("MainWindow", "Segmentation "))
        self.pushButton_8.setText(_translate("MainWindow", "open"))
        self.pushButton_9.setText(_translate("MainWindow", "erosion"))
        self.pushButton_10.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_11.setText(_translate("MainWindow", "PushButton"))
