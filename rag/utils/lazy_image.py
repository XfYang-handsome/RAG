"""RAGFlow rag.utils.lazy_image 的轻量替代实现。"""

from io import BytesIO

from PIL import Image


class LazyImage:
    """惰性图片容器，保存原始字节/图片数据，按需解码。"""

    def __init__(self, image_blobs=None):
        self._image_blobs = image_blobs or []
        self._images = None

    @property
    def image_blobs(self):
        return self._image_blobs

    @property
    def images(self):
        if self._images is None:
            self._images = [self._decode(b) for b in self._image_blobs]
        return self._images

    @staticmethod
    def _decode(b):
        if isinstance(b, Image.Image):
            return b
        if isinstance(b, (bytes, bytearray, memoryview)):
            return Image.open(BytesIO(bytes(b))).convert("RGB")
        return b

    def to_pil(self):
        imgs = self.images
        return imgs[0] if imgs else None

    def __array__(self, dtype=None):
        import numpy as np

        pil = self.to_pil()
        if pil is None:
            return np.array([], dtype=dtype)
        return np.array(pil, dtype=dtype)


def ensure_pil_image(img):
    """返回 PIL.Image（仅当输入本身是 PIL.Image 或 LazyImage）。

    与原版 RAGFlow 语义一致：numpy 数组（含归一化后的 float32）不做转换，
    直接返回 None，由调用方保持 ndarray 继续处理。
    """
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, LazyImage):
        return img.to_pil()
    return None
