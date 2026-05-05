import contextily as ctx
import xyzservices.providers as providers
from PIL import Image

def fetch_satellite(lat, lon):
    delta = 0.01  # larger area, fewer tiles
    w, s, e, n = lon-delta, lat-delta, lon+delta, lat+delta
    img, _ = ctx.bounds2img(
        w, s, e, n,
        ll=True,
        source=providers.OpenStreetMap.Mapnik,
        zoom=12
    )
    return Image.fromarray(img).convert("RGB")

