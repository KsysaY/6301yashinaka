import asyncio
import aiohttp
import aiofiles
import cv2
import os
import logging
import numpy as np
import random
from .models import Artwork
from metetl.decorators import timer

logger = logging.getLogger("metetl.processor")

class ImageProcessor:
    """Класс, управляющий процессом обработки изображений"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def get_valid_image(self, session, ind, all_ids):
        """Загрузка изображений из общего списка ID"""
        while all_ids:
            current_id = all_ids.pop(random.randrange(len(all_ids)))
            try:
                meta_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{current_id}"
                async with session.get(meta_url, timeout=20) as resp:
                    if resp.status != 200: 
                        logger.debug(f"ID {current_id}: API returned status {resp.status}")
                        continue
                    js = await resp.json()
                    img_url = js.get('primaryImage')
                    
                    if not img_url: 
                        logger.debug(f"Image {ind}: ID {current_id} photo missing in metadata. Searching next")
                        continue 

                logger.info(f"Downloading image {ind} started (ID {current_id})")
                orig_path = os.path.join(self.output_dir, f"{ind}_{current_id}_original.png")

                async with session.get(img_url, timeout=60) as resp:
                    if resp.status != 200: 
                        logger.error(f"Image {ind}: Failed to download image file. Status {resp.status}")
                        continue
                    
                    async with aiofiles.open(orig_path, mode='wb') as f:
                        async for chunk in resp.content.iter_chunked(65536):
                            await f.write(chunk)

                logger.info(f"Downloading image {ind} finished")
                return (ind, current_id, orig_path, self.output_dir)
            
            except Exception as e:
                logger.error(f"Unexpected error for Image {ind} (ID {current_id}): {e}")
                continue
        return None

    async def download_all(self, items):
        """Превращает список словарей из JSON в список ID"""
        all_ids = [int(item['object_id']) for item in items]
        
        async with aiohttp.ClientSession() as session:
            count = len(items)
            tasks = [self.get_valid_image(session, i, all_ids) for i in range(1, count + 1)]
            
            results = []
            for res in asyncio.as_completed(tasks):
                result = await res
                if result:
                    results.append(result)
            
            return results

    @staticmethod
    def process_worker(data):
        """Обработка в параллельных процессах"""
        from metetl.logging_config import setup_logging
        setup_logging() 
        worker_logger = logging.getLogger("metetl.worker")

        ind, painting_id, img_path, output_dir = data
        pid = os.getpid()
        
        worker_logger.info(f"Convolution for image {ind} started (PID {pid})")
        
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                worker_logger.error(f"Could not read file {img_path}")
                return False
            
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
                
            worker_logger.info(f"Convolution for image {ind} finished")
            return True
        except Exception as e:
            worker_logger.error(f"Error in worker for image {ind}: {e}")
            return False