#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from hycoclip.config import LazyCall as L
from hycoclip.evaluation.classification import ZeroShotClassificationEvaluator


evaluator = L(ZeroShotClassificationEvaluator)(
    datasets_and_prompts={
        # "imagenet": [
        #     "i took a picture : itap of a {}.",
        #     "pics : a bad photo of the {}.",
        #     "pics : a origami {}.",
        #     "pics : a photo of the large {}.",
        #     "pics : a {} in a video game.",
        #     "pics : art of the {}.",
        #     "pics : a photo of the small {}.",
        # ],
        # "cars": [
        #     "a photo of a {}.",
        #     "a photo of the {}.",
        #     "a photo of my {}.",
        #     "i love my {}!",
        #     "a photo of my dirty {}.",
        #     "a photo of my clean {}.",
        #     "a photo of my new {}.",
        #     "a photo of my old {}.",
        # ],
        # SUCCESS
        # "food101": [
        #     "food : {}.",
        #     "food porn : {}.",
        # ],
        # "cifar10": [
        #     "a photo of a {}.",
        #     "a blurry photo of a {}.",
        #     "a black and white photo of a {}.",
        #     "a low contrast photo of a {}.",
        #     "a high contrast photo of a {}.",
        #     "a bad photo of a {}.",
        #     "a good photo of a {}.",
        #     "a photo of a small {}.",
        #     "a photo of a big {}.",
        #     "a photo of the {}.",
        #     "a blurry photo of the {}.",
        #     "a black and white photo of the {}.",
        #     "a low contrast photo of the {}.",
        #     "a high contrast photo of the {}.",
        #     "a bad photo of the {}.",
        #     "a good photo of the {}.",
        #     "a photo of the small {}.",
        #     "a photo of the big {}.",
        # ],
        # "cifar100": [
        #     "a photo of a {}.",
        #     "a blurry photo of a {}.",
        #     "a black and white photo of a {}.",
        #     "a low contrast photo of a {}.",
        #     "a high contrast photo of a {}.",
        #     "a bad photo of a {}.",
        #     "a good photo of a {}.",
        #     "a photo of a small {}.",
        #     "a photo of a big {}.",
        #     "a photo of the {}.",
        #     "a blurry photo of the {}.",
        #     "a black and white photo of the {}.",
        #     "a low contrast photo of the {}.",
        #     "a high contrast photo of the {}.",
        #     "a bad photo of the {}.",
        #     "a good photo of the {}.",
        #     "a photo of the small {}.",
        #     "a photo of the big {}.",
        # ],
        # FAILED
        # "cub2011": [
        #     "bird pics : {}.",
        #     "birding : {}.",
        #     "birds : {}.",
        #     "bird photography : {}.",
        # ],
        # FAILED
        # "sun397": [
        #     "a photo of a {}.",
        #     "a photo of the {}.",
        # ],
        # SUCCESS
        # "aircraft": [
        #     "a photo of a {}, a type of aircraft.",
        #     "a photo of the {}, a type of aircraft.",
        # ],
        # "dtd": [
        #     "pics : {} texture.",
        #     "pics : {} pattern.",
        #     "pics : {} thing.",
        #     "pics : this {} texture.",
        #     "pics : this {} pattern.",
        #     "pics : this {} thing.",
        # ],
        # "pets": [
        #     "a photo of a {}, a type of pet.",
        # ],
        # FAILED
        # "caltech101": [
        #     "a photo of a {}.",
        #     "a painting of a {}.",
        #     "a plastic {}.",
        #     "a sculpture of a {}.",
        #     "a sketch of a {}.",
        #     "a tattoo of a {}.",
        #     "a toy {}.",
        #     "a rendition of a {}.",
        #     "a embroidered {}.",
        #     "a cartoon {}.",
        #     "a {} in a video game.",
        #     "a plushie {}.",
        #     "a origami {}.",
        #     "art of a {}.",
        #     "graffiti of a {}.",
        #     "a drawing of a {}.",
        #     "a doodle of a {}.",
        #     "a photo of the {}.",
        #     "a painting of the {}.",
        #     "the plastic {}.",
        #     "a sculpture of the {}.",
        #     "a sketch of the {}.",
        #     "a tattoo of the {}.",
        #     "the toy {}.",
        #     "a rendition of the {}.",
        #     "the embroidered {}.",
        #     "the cartoon {}.",
        #     "the {} in a video game.",
        #     "the plushie {}.",
        #     "the origami {}.",
        #     "art of the {}.",
        #     "graffiti of the {}.",
        #     "a drawing of the {}.",
        #     "a doodle of the {}.",
        # ],
        # SUCCESS
        # "flowers102": [
        #     "flowers : {}.",
        # ],
        # "stl10": [
        #     "a photo of a {}.",
        #     "a photo of the {}.",
        # ],
        # "eurosat": [
        #     "a centered satellite photo of {}.",
        #     "a centered satellite photo of a {}.",
        #     "a centered satellite photo of the {}.",
        # ],
        # FAILED
        # "resisc45": [
        #     "satellite imagery of {}.",
        #     "aerial imagery of {}.",
        #     "satellite photo of {}.",
        #     "aerial photo of {}.",
        #     "satellite view of {}.",
        #     "aerial view of {}.",
        #     "satellite imagery of a {}.",
        #     "aerial imagery of a {}.",
        #     "satellite photo of a {}.",
        #     "aerial photo of a {}.",
        #     "satellite view of a {}.",
        #     "aerial view of a {}.",
        #     "satellite imagery of the {}.",
        #     "aerial imagery of the {}.",
        #     "satellite photo of the {}.",
        #     "aerial photo of the {}.",
        #     "satellite view of the {}.",
        #     "aerial view of the {}.",
        # ],
        # SUCCESS
        # "country211": [
        #     "a photo i took in {}.",
        #     "a photo i took while visiting {}.",
        #     "a photo from my home country of {}.",
        #     "a photo from my visit to {}.",
        #     "a photo showing the country of {}.",
        # ],
        # SUCCESS
        # "mnist": [
        #     'a photo of the number: "{}".',
        # ],
        # SUCCESS
        # "clevr": [
        #     "a photo of {} objects.",
        # ],
        "pcam": [
            "this is a photo of {}.",
        ],
        "sst2": [
            "a {} review of a movie.",
        ],
    },
    data_dir="datasets/eval",
    image_size=224
)
