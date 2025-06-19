# -*- coding: utf-8 -*-
# author: haroldchen0414

from imutils.perspective import four_point_transform
from PIL import Image, ImageOps
from paddleocr import PaddleOCR
from imutils import paths
import numpy as np
import imutils
import csv
import cv2
import re
import os

class BeanCardDetector:
    def __init__(self):
        self.ocrDetector = PaddleOCR(use_textline_orientation=True, lang="ch")
        self.width = 500
            
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        # 手机图片会在图片的EXIF元数据存储Orientaion标签表示手机是横拍, 竖排还是倒置拍
        # 分析图片exif信息确保图片的旋转是正确的
        image = ImageOps.exif_transpose(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = imutils.resize(image, width=self.width)

        return image

    def find_card(self, image_path, debug=False):
        image = self.preprocess_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        if debug:
            cv2.imshow("image", image)
            cv2.imshow("Edged", edged)
            cv2.waitKey(0)
        
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cardCnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                cardCnt = approx
                break
        
        if cardCnt is None:
            raise Exception("找不到卡片轮廓, 请调整边缘检测参数")

        card = four_point_transform(image, cardCnt.reshape(4, 2))

        if debug:
            output = image.copy()
            cv2.drawContours(output, [cardCnt], -1, (0, 255, 0), 2)
            cv2.imshow("Card", output)
            cv2.imshow("Transform", imutils.resize(card, width=500))
            cv2.waitKey(0)
        
        return card
    
    def ocr(self, image_path):
        cardImage = self.find_card(image_path)
        result = self.ocrDetector.ocr(cardImage, cls=True)
        text = [wordInfo[1][0] for line in result for wordInfo in line]
        print(text, len(text))
        return text
    
    def write_csv(self, data, csv_path="coffee_data.csv"):
        headers = ["名字", "品种", "处理法", "海拔", "产区", "庄园", "庄园主", "风味", "测评分数", "品牌"]
        fileExists = os.path.exists(csv_path)

        with open(csv_path, "a" if fileExists else "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)

            if not fileExists:
                writer.writerow(headers)

            writer.writerow(data)

if __name__ == "__main__":
    detector = BeanCardDetector()
    imagePaths = list(paths.list_images("./"))

    for imagePath in imagePaths:
        content = detector.ocr(imagePath)

        name = content[2]
        variety = content[5]
        method = content[7]
        altitude = content[9]
        area = content[11]
        estate = content[12]
        estateOwner = content[14]
        flavor = content[15]
        score = content[16]
        brand = content[18]

        data = [name, variety, method, altitude, area, estate, estateOwner, flavor, score, brand]
        detector.write_csv(data)