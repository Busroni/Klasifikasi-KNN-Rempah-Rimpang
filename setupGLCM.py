from skimage.feature import graycoprops


class setGLCM:
    def __init__(self, matrix_coocurrence):
        self.matrix_coocurrence = matrix_coocurrence

    def contrast_feature(self):
        contrast = graycoprops(self.matrix_coocurrence, 'contrast')
        return contrast

    def homogeneity_feature(self):
        homogeneity = graycoprops(self.matrix_coocurrence, 'homogeneity')
        return homogeneity

    def energy_feature(self):
        energy = graycoprops(self.matrix_coocurrence, 'energy')
        return energy

    def correlation_feature(self):
        correlation = graycoprops(self.matrix_coocurrence, 'correlation')
        return correlation

