from PIL import Image
from numpy import asarray
import base64
import io


def base64ToArray(data):
    decoded = base64.b64decode(data)
    image = Image.open(io.BytesIO(decoded))
    return asarray(image)