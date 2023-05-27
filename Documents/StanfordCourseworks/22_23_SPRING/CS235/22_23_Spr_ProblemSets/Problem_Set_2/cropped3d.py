
# Lint as: python3
"""Breast Cacner Cropped and Enhanced"""

import datasets
import pandas as pd

logger = datasets.logging.get_logger(__name__)



_URL = {
    "train": "https://huggingface.co/datasets/yanmiamin/breastCancerEnhancedTrain/resolve/main/cropped3c_train.tar.gz",
}

_URL_labels = {
    "https://huggingface.co/datasets/yanmiamin/breastCancerEnhancedTrain/blob/main/labels_small.csv"
}

labels = pd.read_csv(_URL_labels)

class breastCancer(datasets.GeneratorBasedBuilder):
    
    """cropped and enhanced mammographies"""

    def _info(self):
        return datasets.DatasetInfo(
    
            features=datasets.Features(
                {
                    "labels": datasets.Value("int"),
                    "image": datasets.Image(),
    
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/yanmiamin/breastCancerEnhancedTrain/",
    
        )

    def _split_generators(self, dl_manager):
        
        downloaded_files = dl_manager.download_and_extract(_URL)
        image_iters = dl_manager.iter_archive(downloaded_files)
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"images": image_iters}),
        ]

    def _generate_examples(self, images):
        """This function returns the examples in the raw (text) form."""

        idx = 0
        for filepath, image in images:
            name = filepath.split('/')[-1]
            name = name.split('.')[0]
            label = labels.loc[labels['name'] == name, 'label'].iloc[0]
            yield idx, {
                'image_file_path': filepath,
                'image': image,
                'labels': label
            }
            idx += 1