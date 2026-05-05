import numpy as np

# Simple interpretable proxies from ViT embeddings
def vit_to_indicators(vit_features):
    """
    vit_features: [N,768]
    Returns numeric indicator scores per district
    """

    # Normalize features
    f = (vit_features - vit_features.mean(axis=0)) / (vit_features.std(axis=0) + 1e-6)

    # Use fixed channel groups as proxy indicators
    built_up = f[:, :200].mean(axis=1)
    vegetation = f[:, 200:400].mean(axis=1)
    roads = f[:, 400:600].mean(axis=1)
    nightlight = f[:, 600:768].mean(axis=1)

    # Min-max scale to 0–1
    def scale(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    return {
        "built_up": scale(built_up),
        "vegetation": scale(vegetation),
        "roads": scale(roads),
        "nightlight": scale(nightlight)
    }
