import easyocr
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm


def processSingleImage(image, reader, desiredText):
    result = reader.readtext(image, detail=0)
    if desiredText in result:
        return image
    else:
        return None


def update_progress_bar(pbar, single_result, agregatedList):
    pbar.update(1)
    agregatedList.append(single_result)


def parallelProcess(argumentList, workers):
    try:
        set_start_method("spawn")
    except RuntimeError as e:
        print(e)
        raise

    mappedValues = []
    with Pool(processes=workers) as pool:
        with tqdm(total=len(argumentList), desc="Processed files") as pbar:
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

    # Only return non NULL values
    return [image for image in mappedValues if image]


if __name__ == "__main__":
    reader = easyocr.Reader(["es"])
    baseDir = Path("/home/andres/workshop/fb/gallery-dl/facebook/Ruta de las Iglesias")
    workers = 2
    desiredText = '382'

    fileNames = []
    for file in baseDir.rglob("*.jpg"):
        if file.is_file():
            fileNames.append(str(file))

    argumentList = [(image, reader, desiredText) for image in fileNames]

    result = parallelProcess(argumentList, workers)

    with open("results.csv", "w") as f:
        f.write("\n".join(result))
