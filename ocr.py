import easyocr
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
import argparse
import csv


# One reader per worker process, created by initWorker
reader = None


def initWorker():
    global reader
    reader = easyocr.Reader(['es'])


def processSingleImage(image, desiredTexts):
    result = reader.readtext(image, detail=0, batch_size=1000)
    # One (bib, image) pair per bib found in the image
    return [(text, image) for text in desiredTexts if text in result]


def update_progress_bar(pbar, single_result, agregatedList):
    pbar.update(1)
    agregatedList.append(single_result)


def parallelProcess(argumentList, workers):
    try:
        set_start_method('spawn')
    except RuntimeError as e:
        print(e)
        raise

    mappedValues = []
    with Pool(processes=workers, initializer=initWorker) as pool:
        with tqdm(total=len(argumentList), desc='Processed files') as pbar:
            for args in argumentList:
                pool.apply_async(
                    processSingleImage,
                    args=args,
                    callback=lambda single_result: update_progress_bar(
                        pbar, single_result, mappedValues
                    ),
                )
            pool.close()
            pool.join()

    # Flatten the per-image match lists
    return [match for matches in mappedValues for match in matches]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--workers', required=True, type=int)
    parser.add_argument('--desired-text', required=True, nargs='+')
    args = parser.parse_args()

    baseDir = Path(args.root_dir)
    workers = args.workers
    desiredTexts = args.desired_text

    fileNames = []
    for file in baseDir.rglob('*.jpg'):
        if file.is_file():
            fileNames.append(str(file))

    argumentList = [(image, desiredTexts) for image in fileNames]

    result = parallelProcess(argumentList, workers)

    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bib', 'image'])
        writer.writerows(sorted(result))
