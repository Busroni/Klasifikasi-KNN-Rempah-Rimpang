from PIL import Image


class SizeCrop:
    def __init__(self, image):
        self.image = image

    def proses(self):
        newImage = './static/img_proses/sizecrop.jpg'
        im = Image.open(self.image)
        width, height = im.size
        print(width, height)
        im1 = im
        # Setting the points for cropped image
        # CROPPING IMAGE
        if (width > 256 and height > 256):
            cropHorizontal = (width - 256)/2
            cropVertical = (height - 256)/2
            left = cropHorizontal
            top = cropVertical
            right = 256 + cropHorizontal
            bottom = 256 + cropVertical
            im1 = im.crop((left, top, right, bottom))
            im1.save(newImage)
        elif (width > 256 and height == 256):
            cropHorizontal = (width - 256)/2
            left = cropHorizontal
            top = 0
            right = 256 + cropHorizontal
            bottom = 0
            im1 = im.crop((left, top, right, bottom))
            im1.save(newImage)
        elif (width == 256 and height > 256):
            cropVertical = (height - 256)/2
            left = 0
            top = cropVertical
            right = 0
            bottom = 256 + cropVertical
            im1 = im.crop((left, top, right, bottom))
            im1.save(newImage)
        elif (width < 256 or height < 256):
            # RESIZING IMAGE
            im1 = im.resize((256, 256))
            im1.save(newImage)
        else:
            im1 = im
            im1.save(newImage)
        return newImage