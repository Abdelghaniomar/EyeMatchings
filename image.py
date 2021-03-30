from app2 import *
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication,QFileDialog
import os
import sys
class My_Application(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.checkPath)
        self.ui.pushButton2.clicked.connect(self.file_open)
        print(self.checkPath())
    def checkPath(self):
        image_path = self.ui.lineEdit.text()
        if os.path.isfile(image_path):
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QPixmap(image_path)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.graphicsView.setScene(scene)



    def file_open(self):
             name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')

             print(str(name)[2:-19])
             self.ui.lineEdit.setText(str(name)[2:-19])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    class_instance = My_Application()
    class_instance.show()
    sys.exit(app.exec_())