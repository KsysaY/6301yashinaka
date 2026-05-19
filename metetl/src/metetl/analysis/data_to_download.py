import csv
import json
import logging

logger = logging.getLogger("metetl.prepare")

def prepare_metadata(csv_path, output_json, limit):
    paintings = []
    logger.debug(f"Opening CSV: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Classification') == 'Paintings':
                    paintings.append({
                        'object_id': row['Object ID'],
                        'title': row['Title']
                    })
                
                if limit is not None and len(paintings) >= limit:
                    logger.debug(f"Limit reached: {limit} items.")
                    break
                    
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(paintings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Found and saved {len(paintings)} items to {output_json}")
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")