import argparse
import asyncio
import json
import logging
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor

from .logging_config import setup_logging
from .analysis.data_to_download import prepare_metadata
from .analysis.aggregations import run_analysis
from .images.processing import ImageProcessor

def main():
    setup_logging()
    logger = logging.getLogger("metetl.cli")

    parser = argparse.ArgumentParser(
        prog='metetl',
        description='Скачивание, обработка и анализ данных'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды', required=True)

    prep_p = subparsers.add_parser('prepare', help='Подготовка JSON с метаданными')
    prep_p.add_argument('--csv', required=True, help='Путь к исходному CSV файлу')
    prep_p.add_argument('--output', required=True, help='Путь для сохранения JSON')
    prep_p.add_argument('--num', type=int, default=None, help='Количество метаданных')

    proc_p = subparsers.add_parser('process', help='Запуск скачивания и обработки')
    proc_p.add_argument('--input', required=True, help='Путь к JSON файлу с отобранными ID')
    proc_p.add_argument('--output', required=True, help='Директория для сохранения изображений')
    proc_p.add_argument('--num', type=int, default=5, help='Количество изображений для обработки')

    ana_p = subparsers.add_parser('analyze', help='Анализ датасета и построение графиков')
    ana_p.add_argument('--csv', required=True, help='Путь к CSV файлу')
    ana_p.add_argument('--output-dir', required=True, help='Директория для сохранения графиков')

    args = parser.parse_args()

    if args.command == 'prepare':
        logger.info(f"Starting preparation: {args.csv} - {args.output} (limit: {args.num})")
        prepare_metadata(args.csv, args.output, args.num)
        logger.info("Preparation finished.")

    elif args.command == 'process':
        if not os.path.exists(args.input):
            logger.error(f"Input file {args.input} not found!")
            return

        with open(args.input, 'r', encoding='utf-8') as f:
            items = json.load(f)
        
        num = min(len(items), args.num)
        selected_items = random.sample(items, num)

        processor = ImageProcessor(output_dir=args.output)
        
        logger.info(f"Starting process for {len(selected_items)} images")
        start_t = time.time()

        downloaded_data = asyncio.run(processor.download_all(selected_items))

        if not downloaded_data:
            logger.error("No images downloaded.")
            return

        logger.info(f"Starting parallel processing on CPU cores")
        with ProcessPoolExecutor() as executor:
            executor.map(ImageProcessor.process_worker, downloaded_data)

        logger.info(f"Process finished in {time.time() - start_t:.2f}s")

    elif args.command == 'analyze':
        logger.info(f"Starting analysis of {args.csv}")
        run_analysis(args.csv, args.output_dir)


if __name__ == "__main__":
    main()