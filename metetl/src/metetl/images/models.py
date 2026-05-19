import numpy as np
import logging

logger = logging.getLogger("metetl.artwork")

class Artwork:
    def __init__(self, image: np.ndarray, metadata: dict = None):
        self._image = image
        self._metadata = metadata if metadata is not None else {}

    @property
    def title(self) -> str: 
        return self._metadata.get('title', 'Unknown')
    
    @property
    def artist(self) -> str: 
        return self._metadata.get('artist', 'Unknown')

    def _MyConvolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Собственная свёртка"""
        kh, kw = kernel.shape
        h, w = image.shape[:2]
        pad_h, pad_w = kh // 2, kw // 2
        is_rgb = len(image.shape) == 3
        
        if is_rgb:
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge').astype(np.float32)
            result = np.zeros((h, w, 3), dtype=np.float32)
        else:
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge').astype(np.float32)
            result = np.zeros((h, w), dtype=np.float32)

        for i in range(kh):
            for j in range(kw):
                ki = kernel[i, j]
                if ki == 0: continue
                result += padded[i : i + h, j : j + w] * ki
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def MyConvolution(self, kernel: np.ndarray) -> np.ndarray:
        return self._MyConvolution(self._image, kernel)
        
    def _MyGrayscale(self) -> np.ndarray:
        gray = 0.299 * self._image[:, :, 0] + 0.587 * self._image[:, :, 1] + 0.114 * self._image[:, :, 2]
        return gray.astype(np.uint8)
    
    def MyGrayscale(self) -> np.ndarray:
        return self._MyGrayscale()
    
    def _MyBlur(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Cглаживание (применение оператора Гаусса) (Собственная реализация)"""
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        x = np.arange(size) - center
        x = np.stack([x]*size)
        y = np.matrix.transpose(x)
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        kernel = kernel / np.sum(kernel)
        return self.MyConvolution(kernel)
    
    def MyBlur(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Cглаживание (применение оператора Гаусса)"""
        return self._MyBlur(size, sigma)
    
    def _MySobel(self) -> np.ndarray:
        """Выделение границ (применение оператора Собеля) (собственная реализация)"""

        kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
    
        kernel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)
        
        if len(self._image.shape) == 3:
            gray = self.MyGrayscale()
        else:
            gray = self._image

        grad_x = self._MyConvolution(gray, kernel_x)
        grad_y = self._MyConvolution(gray, kernel_y)
        
        approximate_value = np.sqrt(grad_x.astype(np.float32)**2 + grad_y.astype(np.float32)**2)
        approximate_value = np.clip(approximate_value, 0, 255)
        
        return approximate_value.astype(np.uint8)
    
    def MySobel(self) -> np.ndarray:
        return self._MySobel()
    
    def __add__(self, other: 'Artwork') -> 'Artwork':
        """Перегрузка метода сложения изображений"""
        if self._image.shape != other._image.shape:
            return None
        
        result_image = np.clip(self._image + other._image, 0, 255)
        
        return Artwork(result_image, self._metadata)
    
    def __str__(self) -> str:
        """Перегрузка метода преобразования в строку"""
        return (f"Artwork: {self.title()}\n  Artist: {self.artist()}\n")