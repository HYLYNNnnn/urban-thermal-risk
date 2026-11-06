"""Read the point identity carried by the recovered image filenames."""
from pathlib import Path
import math
import re

STREET_NAME = re.compile(r'^(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$')
FISHEYE_NAME = re.compile(r'^(\d+)_fisheye$', re.IGNORECASE)


def image_metadata(path: Path, root: Path) -> dict:
    path, root = Path(path).resolve(), Path(root).resolve()
    relative = path.relative_to(root)
    match = STREET_NAME.fullmatch(path.stem)
    longitude = latitude = heading = None
    if match:
        point_id = str(int(match[1]))
        longitude, latitude, heading = map(float, match.groups()[1:])
        if not all(map(math.isfinite, [longitude, latitude, heading])) or not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError(f'Invalid coordinates in image name: {relative}')
        heading %= 360
    else:
        fish = FISHEYE_NAME.fullmatch(path.stem)
        if fish:
            point_id = str(int(fish[1]))
        else:
            raise ValueError(f'Use POINT_LONGITUDE_LATITUDE_HEADING.ext or POINT_fisheye.ext: {relative}')
    if int(point_id) < 1:
        raise ValueError('Point IDs must be positive integers.')
    if path.parent.name.isdigit() and int(path.parent.name) != int(point_id):
        raise ValueError(f'Image point ID disagrees with its folder: {relative}')
    return {'point_id': point_id, 'image': relative.as_posix(), 'longitude': longitude,
            'latitude': latitude, 'heading': heading}
