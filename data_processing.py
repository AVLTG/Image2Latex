import re
import json
import multiprocessing as mp
from utils import strokes_to_image
from config import DETEXIFY_DUMP_PATH, IMAGE_SIZE, NUM_WORKERS, CHUNK_SIZE

def parse_chunk(chunk):
    samples = []
    for line in chunk:
        if line.strip() == '':
            continue
        parts = line.split('\t')
        if len(parts) == 3:
            sample_id, key, strokes = parts
            strokes = json.loads(strokes)
            img = strokes_to_image(strokes, img_size=IMAGE_SIZE)
            samples.append((img, key))
    return samples

def parse_dump_parallel():
    # Read the file and extract the data lines
    with open(DETEXIFY_DUMP_PATH, 'r') as f:
        content = f.read()
        pattern = re.compile(r'COPY samples \(id, key, strokes\) FROM stdin;\n(.*?)\.\n', re.DOTALL)
        matches = pattern.search(content)
        if matches:
            data_lines = matches.group(1).split('\n')
        else:
            return []

    # Split data into chunks for parallel processing
    chunks = [data_lines[i:i + CHUNK_SIZE] for i in range(0, len(data_lines), CHUNK_SIZE)]

    # Use multiprocessing Pool to parse chunks in parallel
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(parse_chunk, chunks)

    # Flatten the list of results
    samples = [item for sublist in results for item in sublist]
    return samples

def load_data():
    samples = parse_dump_parallel()
    images, labels = zip(*samples)
    return images, labels
