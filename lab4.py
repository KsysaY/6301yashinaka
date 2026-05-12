#!/usr/bin/env python3
import asyncio
import aiohttp
import aiofiles
import os
import time
import argparse
import cv2
import numpy as np
import csv
import random
from concurrent.futures import ProcessPoolExecutor

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
    
#Декоратор, замеряющий время выполнения функции    
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} - {end - start:.4f} сек")
        return result
    return wrapper

class ImageProcessor:
    """Класс, управляющий процессом обработки изображений"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def get_valid_image(self, session, ind, all_ids):
        """Загрузка изображений"""
        while all_ids:
            current_id = all_ids.pop(random.randrange(len(all_ids)))
            try:
                meta_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{current_id}"
                async with session.get(meta_url, timeout=40) as resp:
                    if resp.status != 200: 
                        continue
                    js = await resp.json()
                    img_url = js.get('primaryImage')
                    
                    if not img_url: 
                        print(f"Image {ind}: ID {current_id} photo missing. Searching next")
                        continue 

                print(f"Downloading image {ind} started")

                async with session.get(img_url, timeout=40) as resp:
                    if resp.status != 200: 
                        continue
                    img_bytes = await resp.read()

                orig_path = os.path.join(self.output_dir, f"{ind}_{current_id}_original.png")
                async with aiofiles.open(orig_path, mode='wb') as f:
                    await f.write(img_bytes)

                print(f"Downloading image {ind} finished")
                return (ind, current_id, img_bytes, self.output_dir)
            except Exception:
                continue
        return None

    async def run_downloads(self, count):
        """Запуск параллельных загрузок"""
        async with aiohttp.ClientSession() as session:
            search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=True&q=Paintings"

            async with session.get(search_url) as resp:
                if resp.status != 200:
                    print("Could not access Met Museum API")
                    return []
                search_data = await resp.json()
                all_ids = search_data.get('objectIDs', [])
                
            if not all_ids:
                print("No ID found")
                return []

            tasks = [self.get_valid_image(session, i, all_ids) for i in range(1, count + 1)]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]

    @staticmethod
    def process_worker(data):
        """Обработка в параллельных процессах"""
        ind, painting_id, img_bytes, output_dir = data
        pid = os.getpid()
        
        print(f"Convolution for image {ind} started (PID {pid})")
        
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        artwork = Artwork(img_rgb)
        kernel = np.array([[-2, -1, 0], 
                               [-1, 1, 1], 
                               [0, 1, 2]], dtype=np.float32)
        
        jobs = [
            (artwork.MyConvolution(kernel), "conv.png"),
            (artwork.MyGrayscale(), "gray.png"),
            (artwork.MyBlur(7, 1.5), "blur.png"),
            (artwork.MySobel(), "sobel.png")
        ]
        
        for res_img, suffix in jobs:
            fname = f"{ind}_{painting_id}_{suffix}"
            out = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR) if len(res_img.shape) == 3 else res_img
            cv2.imwrite(os.path.join(output_dir, fname), out)
            
        print(f"Convolution for image {ind} finished")
        return True
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='Количество изображений для скачивания')
    args = parser.parse_args()

    processor = ImageProcessor(output_dir="lab4_output")
    
    start_total = time.time()

    downloaded_data = asyncio.run(processor.run_downloads(args.n))

    if not downloaded_data:
        print("No images downloaded")
        return

    print(f"\nОбработка изображений:")
    with ProcessPoolExecutor() as execut:
        execut.map(ImageProcessor.process_worker, downloaded_data)

    total_time = time.time() - start_total
    print(f"\nОбщее время работы: {total_time:.2f} сек")

if __name__ == "__main__":
    main()