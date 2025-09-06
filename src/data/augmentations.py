import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size, use_imagenet_stats=False):
    """
    Returns the transformations for the training set.
    """
    if use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.696, 0.522, 0.426]
        std = [0.142, 0.135, 0.128]

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.ColorJitter(
            brightness=0.1,   # ±10% brightness
            contrast=0.1,     # ±10% contrast
            saturation=0.1,   # ±10% saturation
            hue=0.05,         # small hue shift
            p=0.5
        ),
        A.Normalize(
            mean=mean,
            std=std,
        ),
        ToTensorV2(),
    ])

def get_val_transforms(img_size, use_imagenet_stats=False):
    """
    Returns the transformations for the validation set.
    """
    if use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.696, 0.522, 0.426]
        std = [0.142, 0.135, 0.128]

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=mean,
            std=std,
        ),
        ToTensorV2(),
    ])

def get_test_transforms(img_size, use_imagenet_stats=False):
    """
    Returns the transformations for the test set.
    """
    if use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.696, 0.522, 0.426]
        std = [0.142, 0.135, 0.128]

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=mean,
            std=std,
        ),
        ToTensorV2(),
    ])




class TwoAugmentTransform:
    """Create two augmented versions of the same image. Used for BYOL."""
    def __init__(self, img_size: int):
        self.t1 = get_train_transforms(img_size)
        self.t2 = get_train_transforms(img_size)

    def __call__(self, image):
        v1 = self.t1(image=image)['image']
        v2 = self.t2(image=image)['image']
        return v1, v2
    

class InferenceTransform:
    """Transform for inference: single view with validation transforms."""
    def __init__(self, img_size: int):
        self.transform = get_val_transforms(img_size)

    def __call__(self, image):
        return self.transform(image=image)['image']