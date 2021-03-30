def eros(self):
    # Read the image for erosion
    i = str(self.ui.lineEdit.text())
    image = cv2.imread(i)
    # def erosion(image,se):
    # Acquire size of the image
    m = image.shape[0]
    n = image.shape[1]
    # Show the image
    plt.imshow(image, cmap="gray")

    # Define the structuring element
    # k= 11,15,45 -Different sizes of the structuring element
    k = 11
    SE = np.ones((k, k), dtype=np.uint8)
    # k= se.shape
    constant = (k - 1) // 2
    # Define new image
    imgErode = np.zeros((m, n), dtype=np.uint8)
    # Erosion without using inbuilt cv2 function for morphology
    for i in range(constant, m - constant):
        for j in range(constant, n - constant):
            temp = image[i - constant:i + constant + 1, j - constant:j + constant + 1]
            product = temp * SE
            imgErode[i, j] = np.min(product)

    plt.imshow(imgErode, cmap="gray")
    cv2.imshow("image erode", imgErode)
    cv2.waitKey(0)
    cv2.imwrite("Eroded3.png", imgErode)

    a = str(self.ui.lineEdit.text())[0:-6]
    a = a + "Eroded3.png"
    if os.path.isfile(a):
        scene = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap(a)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.graphicsView_2.setScene(scene)
##########################################################################################################
