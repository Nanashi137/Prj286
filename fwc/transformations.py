import albumentations as A
import cv2 as cv
import numpy as np
import numpy.typing as npt

from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


def resize_fixed_height(img: npt.NDArray[np.uint8], new_height: int = 105):
    # From the paper: height is fixed to 105 pixels, width is scaled
    # to keep aspect ratio.
    height, width = img.shape[:2]
    new_width = round(new_height * width / height)
    return cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LANCZOS4)

def inverse_pixel(img: npt.NDArray[np.uint8]): 
    return 255 - img


def split_patches_np(img: npt.NDArray[np.uint8], step: int, drop_last: bool):
    height, width = img.shape

    patches = []
    for x in range(0, width, step):
        patches.append(img[0:height, x : x + step])

    # Fixup the last patch instead of dropping it because
    # its width is smaller than 105. When cropping and
    # the patch is smaller than the needed area, it gets filled
    # with black pixels. We should recolor them instead of discarding.
    available_width = width % step
    if available_width != 0:
        if drop_last:
            patches.pop()
        else:
            patches[-1] = np.append(
                patches[-1],
                np.full((height, step - available_width), 255, dtype="uint8"),
                axis=1,
            )

    return patches


class PickRandomPatch(ImageOnlyTransform):
    """
    Pick a random 105x105 patch.
    """

    def __init__(
        self,
        constrained_patches=True,
        drop_last=False,
        always_apply=False,
        p=1.0,
    ) -> None:
        super(PickRandomPatch, self).__init__(always_apply, p)
        self.constrained_patches = constrained_patches
        self.drop_last = drop_last

    def apply(self, img, **params):
        # If patches are constrained, then we split the
        # image into 105x105 boxes and pick one of the boxes.
        # Otherwise, we pick a random start coordinate and
        # build a box from there.

        _, width = img.shape
        if width <= 105:
            return np.append(
                img,
                np.full((105, 105 - width), 255, dtype="uint8"),
                axis=1,
            )

        if not self.constrained_patches:
            start_x = np.random.randint(0, width - 105)
            return img[0:105, start_x : start_x + 105]

        patches = split_patches_np(img, 105, self.drop_last)
        patch_index = np.random.randint(0, len(patches))
        return patches[patch_index]


class VariableAspectRatio(ImageOnlyTransform):
    """
    The image, with heigh fixed, is squeezed in width by a random
    ratio, drawn from a uniform distribution between a range.
    """

    def __init__(self, ratio_range, always_apply=False, p=1.0) -> None:
        super(VariableAspectRatio, self).__init__(always_apply, p)
        self.ratio_range = ratio_range

    def apply(self, img, **params):
        height, width = img.shape
        ratio = np.random.uniform(low=self.ratio_range[0], high=self.ratio_range[1])
        new_width = round(width * ratio)
        squeezed = cv.resize(img, (new_width, height), cv.INTER_LANCZOS4)
        return np.array(squeezed)

    def get_transform_init_args_names(self):
        return ("ratio_range",)


class Squeezing(ImageOnlyTransform):
    """
    Add a "squeezing" operation = "we introduce a squeezing operation, that
    scales the width of the height-normalized image to be of a constant ratio relative
    to the height (2.5 in all our experiments). Note that the squeezing operation is
    equivalent to producing long rectangular input patches."
    """

    def __init__(self, squeeze_ratio, always_apply=False, p=1.0) -> None:
        super(Squeezing, self).__init__(always_apply, p)
        self.squeeze_ratio = squeeze_ratio

    def apply(self, img, **params):
        height, width = img.shape
        new_width = round(height * self.squeeze_ratio)
        squeezed = cv.resize(img, (new_width, height), cv.INTER_LANCZOS4)
        return np.array(squeezed, dtype="uint8")

    def get_transform_init_args_names(self):
        return ("squeeze_ratio",)


class ResizeHeight(ImageOnlyTransform):
    """
    Resize the image height keeping the aspect ratio.
    """

    def __init__(self, target_height: int, always_apply=False, p=1.0) -> None:
        super(ResizeHeight, self).__init__(always_apply, p)
        self.target_height = target_height

    def apply(self, img, **params):
        resized = resize_fixed_height(img, self.target_height)
        return np.array(resized)

class Inverse(ImageOnlyTransform):
    """
    Inverse pixels value in the image
    """
    def __init__(self, alway_apply=False, p=0.4) -> None: 
        super(Inverse, self).__init__(alway_apply, p)

    def apply(self, img, **params):
        iimg = inverse_pixel(img) 
        return iimg 

def get_deepfont_base_augmentations() -> A.Compose:
    return A.Compose(
        [
            # Parameters are taken from the paper.
            A.GaussNoise(var_limit=(3.0, 4.0), p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.5),
            # Images are expected to be grayscale and have a white background, so
            # use the 255 value as a filler.
            A.Affine(rotate=[-3, 3], cval=255, p=0.5),
            A.MultiplicativeNoise(multiplier=(0.3, 0.8), p=0.5),
        ]
    )


def get_deepfont_feature_enhancement() -> A.Compose:
    return A.Sequential(
        [
            # From the deepfont paper, between 5/6 and 7/6.
            VariableAspectRatio(ratio_range=[0.83, 1.17], always_apply=True),
            Squeezing(squeeze_ratio=2.5, always_apply=True),
        ]
    )


def get_random_square_patch() -> A.Compose:
    return A.Sequential(
        [
            PickRandomPatch(constrained_patches=False, always_apply=True),
            A.ToFloat(255, always_apply=True),
            ToTensorV2(),
        ]
    )


def get_deepfont_full_transformation() -> A.Compose:
    return A.Compose(
        [
            A.Sequential(
                [
                    ResizeHeight(target_height=105, always_apply=True),
                    #get_deepfont_base_augmentations(),
                    #get_deepfont_feature_enhancement(),
                    PickRandomPatch(constrained_patches=False, always_apply=True),
                    #A.ToFloat(255, always_apply=True),
                    #ToTensorV2(),
                ]
            ),
        ]
    )


def T1() -> A.Compose:
    return A.Compose(
        [
            A.Sequential(
                [
                    ResizeHeight(target_height=105, always_apply=True),
                    #get_deepfont_base_augmentations(),
                    #get_deepfont_feature_enhancement(),
                    PickRandomPatch(constrained_patches=False, always_apply=True),
                    A.ToFloat(255, always_apply=True),
                    ToTensorV2(always_apply=True),
                ]
            ),
        ]
    )

def get_random_square_patch_augmentation() -> A.Compose:
    return A.Compose(
        [
            ResizeHeight(target_height=105, always_apply=True),
            # Testing still requires VAR and squeezing.
            get_deepfont_feature_enhancement(),
            get_random_square_patch(),
        ]
    )


def get_test_augmentations(squeeze_ratio: float) -> A.Compose:
    return A.Sequential(
        [
            ResizeHeight(target_height=105, always_apply=True),
            Squeezing(squeeze_ratio=squeeze_ratio, always_apply=True),
        ]
    )

def inference_input(img, squeezing_ratio=2.5): 
    t1 = ResizeHeight(target_height= 105, always_apply= True)
    t2 = Squeezing(squeeze_ratio=squeezing_ratio, always_apply=True)
    t3 = PickRandomPatch(constrained_patches=False, always_apply=True)
    t4 = A.ToFloat(255, always_apply=True)
    t5 = ToTensorV2(always_apply=True)


    img = t1.apply(img)
    img = t2.apply(img)
    img = t3.apply(img)
    img = t4.apply(img)
    img = t5.apply(img)

    return img





def IPtrans(img):
    t1 = ResizeHeight(target_height= 105, always_apply= True)
    t2 = PickRandomPatch(constrained_patches=False, always_apply=True)
    t3 = A.ToFloat(255, always_apply=True)
    t4 = ToTensorV2(always_apply=True)

    img = t1.apply(img)
    img = t2.apply(img)
    img = t3.apply(img)
    return t4.apply(img)



def Ptrans(img): 
    t1 = ResizeHeight(target_height= 105, always_apply= True)
    c = np.random.randint(0, 2)
    t1_5 = Inverse(alway_apply=False)
    t2 = PickRandomPatch(constrained_patches=False, always_apply=True)
    t3 = A.ToFloat(255, always_apply=True)
    t4 = ToTensorV2(always_apply=True)

    img = t1.apply(img)
    if c%2 == 0: 
        img = t1_5.apply(img)
    img = t2.apply(img)
    img = t3.apply(img)
    img = t4.apply(img)
    return img