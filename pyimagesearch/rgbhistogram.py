#-*-coding:utf8-*-
import imutils
import cv2
#图像描述符
class RGBHistogram:
	def __init__(self, bins):
		# store the number of bins the histogram will use
		#存储直方图将使用的区间数
		#通过在构造函数中放置相关参数，可以确保为每个图像使用相同的参数。
		self.bins = bins
	
	#describe方法用于“描述”图像并返回特征向量
	def describe(self, image):
		#提取实际的3D RGB直方图（或实际上，BGR，因为OpenCV将图像存储为NumPy数组，但通道的顺序相反）。
		# 假设self.bins是一个三个整数的列表，指定每个通道的区间数。
		hist = cv2.calcHist([image], [0, 1, 2],
			None, self.bins, [0, 256, 0, 256, 0, 256])

		# normalize with OpenCV 2.4
		#根据像素数量对直方图进行标准化非常重要，如果我们使用图像的原始（整数）像素计数，然后将其缩小50％并再次描述它，
		# 我们将为相同的图像提供两个不同的特征向量。在大多数情况下，想要避免这种情况。我们通过将原始整数像素计数转换为实值百分比来获得
		# 尺度不变性。例如，不是说bin＃1中有120个像素，我们会说bin＃1中有20％的像素。同样，通过使用像素计数的百分比而不是原始的整数像素计数，
		# 我们可以确保两个相同的图像（仅在大小上不同）将具有（大致）相同的特征向量
		if imutils.is_cv2():
			hist = cv2.normalize(hist)

		# otherwise normalize with OpenCV 3+
		else:
			hist = cv2.normalize(hist,hist)

		# return out 3D histogram as a flattened array
		#计算3D直方图时，直方图将表示为带有(N, N, N)区间的NumPy数组。为了更容易地计算直方图之间的距离，我们简单地将该直方图展平为具有的形状(N ** 3,)。
		# 示例：当我们实例化RGBHistogram时，每个通道将使用8个bin。没有展平我们的直方图，形状就会如此(8, 8, 8)。但是通过展平它，形状就变成了(512,)
		return hist.flatten()