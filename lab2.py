#Библиотеки
import cv2
import numpy as np
import time
import os
from numpy.lib.stride_tricks import sliding_window_view
import json

#Декоратор, замеряющий время выполнения функции    
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} - {end - start:.4f} сек")
        return result
    return wrapper

class Artwork:
    """Класс, инкапсулирующий изображение и метаданные"""
    def __init__(self, image: np.ndarray, metadata: dict = None):
        self._image = image
        self._metadata = metadata if metadata is not None else {}

    @property
    def image(self) -> np.ndarray:
        """Возвращает изображение"""
        return self._image
    
    @property
    def metadata(self) -> dict:
        """Возвращает метаданные"""
        return self._metadata
    
    @property
    def title(self) -> str:
        """Название"""
        return self._metadata.get('title', 'Unknown')
    
    @property
    def artist(self) -> str:
        """Художник"""
        return self._metadata.get('artist', 'Unknown')
    
    def _MyConvolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Cвёртка c использованием двумерной маски (собственная реализация)"""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        if len(image.shape) == 3:
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
            windows = sliding_window_view(padded, (kh, kw), axis=(0, 1))
            result = np.sum(windows * kernel, axis=(3, 4))
            
            result = np.clip(result, 0, 255)
            return result.astype(np.uint8)
        else:
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            windows = sliding_window_view(padded, (kh, kw))      
            result = np.sum(windows * kernel, axis=(2, 3))
            result = np.clip(result, 0, 255)
            return result.astype(np.uint8)
    
    def MyConvolution(self, kernel: np.ndarray) -> np.ndarray:
        """Cвёртка c использованием двумерной маски (собственная реализация)"""
        return self._MyConvolution(self._image, kernel)
    
    def LibConvolution(self, kernel: np.ndarray) -> np.ndarray:
        """Cвёртка c использованием двумерной маски (библиотечная реализация)"""
        if len(self._image.shape) == 2:
            return cv2.filter2D(self._image, -1, kernel)
        else:
            image_bgr = cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR)
            result_bgr = cv2.filter2D(image_bgr, -1, kernel)
            return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
    def MyGrayscale(self) -> np.ndarray:
        """Приведение цветного изображения к полутоновому (собственная реализация)"""
        return self._MyGrayscale()
    
    def _MyGrayscale(self) -> np.ndarray:
        """Приведение цветного изображения к полутоновому (собственная реализация)"""
        height, width = self._image.shape[0], self._image.shape[1]
        gray = np.zeros((height, width), dtype=np.float32)
        
        gray = 0.299 * self._image[:, :, 0] + 0.587 * self._image[:, :, 1] + 0.114 * self._image[:, :, 2]
        
        return gray.astype(np.uint8)
    
    def LibGrayscale(self) -> np.ndarray:
        """Приведение цветного изображения к полутоновому (библиотечная реализация)"""
        return cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY)
    
    def MyBlur(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Cглаживание (применение оператора Гаусса) (собственная реализация)"""
        return self._MyBlur(size, sigma)
    
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
    
    def LibBlur(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Cглаживание (применение оператора Гаусса) (библиотечная реализация)"""
        if len(self._image.shape) == 2:
            return cv2.GaussianBlur(self._image, (size, size), sigma)
        else:
            image_bgr = cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR)
            result_bgr = cv2.GaussianBlur(image_bgr, (size, size), sigma)
            return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    def MySobel(self) -> np.ndarray:
        """Выделение границ (применение оператора Собеля) (собственная реализация)"""
        return self._MySobel()
    
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
    
    def LibSobel(self) -> np.ndarray:
        """Выделение границ (применение оператора Собеля) (библиотечная реализация)"""
        if len(self._image.shape) == 3:
            gray = self.MyGrayscale()
        else:
            gray = self._image

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        approximate_value = np.sqrt(grad_x**2 + grad_y**2)
        approximate_value = np.clip(approximate_value, 0, 255)
        
        return approximate_value.astype(np.uint8)
    
    def __add__(self, other: 'Artwork') -> 'Artwork':
        """Перегрузка метода сложения изображений"""
        if self._image.shape != other._image.shape:
            return None
        
        result_image = np.clip(self._image + other._image, 0, 255)
        
        return Artwork(result_image, self._metadata)
    
    def __str__(self) -> str:
        """Перегрузка метода преобразования в строку"""
        return (f"Artwork: {self.title()}\n  Artist: {self.artist()}\n")


#Сколько времени с последнего вызова
def timer_metadata(func):
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    start = 0.0
    def wrapper(*args, **kwargs):
        print("BBBBBBBBBBBBBBBBBB")
        nonlocal start
        
        end = time.time()   
        print(f"{func.__name__} - {end - start:.4f} сек")
        start = end
        result = func(*args, **kwargs)
        return result
    return wrapper


class ImageProcessor:
    """Класс, управляющий процессом обработки изображений"""
    def __init__(self, image_path: str):
        self._results = {}
        self._image_path = image_path
        self._artwork = None
        self._load_image()

    
    @property
    def results(self) -> dict:
        return self._results
    
    @property
    def artwork(self) -> Artwork:
        return self._artwork
    
    def _load_image(self):
        """Выгружает изображение и метаданные"""
        img = cv2.imread(self._image_path)
        if img is None:
            print(f"Не удалось загрузить изображение")
            return None
    
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        metadata = None
        json_path = self._image_path.replace('.jpg', '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        self._artwork = Artwork(rgb_image, metadata)
    

    
    def save_result(self, image: np.ndarray, change: str) -> str:
        """Сохранение результата"""
        directory = os.path.dirname(self._image_path)

        basename = os.path.splitext(os.path.basename(self._image_path))[0]
        new_filename = f"{basename}{change}.jpg"
        new_path = os.path.join(directory, new_filename)
        
        cv2.imwrite(new_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        print(f"Сохранено: {new_path}\n")
        return new_path
    
    @timer
    def work_grayscale_lib(self) -> np.ndarray:
        print(" Преобразование в полутоновое:")
        result = self._artwork.LibGrayscale()
        self._results['_lib_gray'] = result
        return result

    def info(self):
        print(f"\nИзображение: {self._image_path}")
        print(f"Размер: {self._artwork.image.shape}")
        print(f"Метаданные: {self._artwork.metadata}\n")

    @timer
    def work_grayscale_my(self) -> np.ndarray:
        print(" Преобразование в полутоновое:")
        result = self._artwork.MyGrayscale()
        self._results['_my_gray'] = result
        return result
    
    @timer
    def work_convolution_lib(self, kernel: np.ndarray = None) -> np.ndarray:
        if kernel is None:
            kernel = np.array([[-2, -1, 0], 
                               [-1, 1, 1], 
                               [0, 1, 2]], dtype=np.float32)
        
        print(" Свертка с ядром:")
        result = self._artwork.LibConvolution(kernel)
        self._results['_lib_conv'] = result
        return result
    
    @timer
    def work_convolution_my(self, kernel: np.ndarray = None) -> np.ndarray:
        if kernel is None:
            kernel = np.array([[-2, -1, 0], 
                               [-1, 1, 1], 
                               [0, 1, 2]], dtype=np.float32)
        
        print(" Свертка с ядром:")
        result = self._artwork.MyConvolution(kernel)
        self._results['_my_conv'] = result
        return result
    
    @timer
    def work_blur_lib(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        print(f" Размытие по Гауссу:")
        result = self._artwork.LibBlur(size, sigma)
        self._results['_lib_blur'] = result
        return result
    
    @timer
    def work_blur_my(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        print(f" Размытие по Гауссу:")
        result = self._artwork.MyBlur(size, sigma)
        self._results['_my_blur'] = result
        return result
    
    @timer
    def work_sobel_lib(self) -> np.ndarray:
        print(" Оператор Собеля:")
        result = self._artwork.LibSobel()
        self._results['_lib_sobel'] = result
        return result
    
    @timer
    def work_sobel_my(self) -> np.ndarray:
        print(" Оператор Собеля:")
        result = self._artwork.MySobel()
        self._results['_my_sobel'] = result
        return result
    
    def process_all(self) -> None:
        #1. Grayscale
        gray = self.work_grayscale_lib()
        self.save_result(gray, "_lib_gray")

        gray = self.work_grayscale_my()
        self.save_result(gray, "_my_gray")
        
        #2. Convolution
        conv = self.work_convolution_lib()
        self.save_result(conv, "_lib_conv")

        #3. Blur
        blur = self.work_blur_lib()
        self.save_result(blur, "_lib_blur")

        blur = self.work_blur_my()
        self.save_result(blur, "_my_blur")
        
        #4. Sobel
        sobel = self.work_sobel_lib()
        self.save_result(sobel, "_lib_sobel")

        sobel = self.work_sobel_my()
        self.save_result(sobel, "_my_sobel")




if __name__ == "__main__":

    ImageProcessor.info = timer_metadata(ImageProcessor.info)
    print(ImageProcessor.info)
    print(type(ImageProcessor.info))
    
    image_path = "paintings/78143.jpg"

    if os.path.exists(image_path):

        a = Artwork.__dict__['artist']
        print(Artwork.__dict__)

        processor = ImageProcessor(image_path)
            
        processor.info()
        processor.info()
            
        processor.process_all()
            
        gray = processor.artwork.MyGrayscale()
        gray_image = Artwork(np.stack([gray]*3, axis=-1))
        processor.info()
            
        result = processor.artwork + gray_image
        processor.save_result(result.image, "_sum")
    else:
        print(f"Файл {image_path} не найден")