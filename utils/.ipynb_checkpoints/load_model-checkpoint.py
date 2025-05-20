from models.resnet_ms import (
    resnet50, resnet50_fc512, resnet50_fc512_ms12_a0d1, resnet50_fc512_ms12_a0d2, resnet50_fc512_ms12_a0d3,
    resnet50_fc512_ms1_a0d1, resnet50_fc512_ms123_a0d1, resnet50_fc512_ms1234_a0d1,
    resnet50_fc512_ms23_a0d1, resnet50_fc512_ms14_a0d1
)

def load_model(cfg):
    """
    Loads the specified backbone model based on the configuration.
    Args:
        cfg: Configuration object with attributes 'backbone.name', 'pretrained_backbone', and 'data.num_classes'.
    Returns:
        Instantiated backbone model.
    Raises:
        ValueError: If the backbone name is not supported.
    """
    if isinstance(cfg.pretrained_backbone, bool):
        pretrained = cfg.pretrained_backbone
    else:
        pretrained = False

    # Map backbone names to their constructor functions
    backbone_constructors = {
        "resnet50": resnet50,
        "resnet50_fc512": resnet50_fc512,
        "resnet50_fc512_ms12_a0d1": resnet50_fc512_ms12_a0d1,
        "resnet50_fc512_ms12_a0d2": resnet50_fc512_ms12_a0d2,
        "resnet50_fc512_ms12_a0d3": resnet50_fc512_ms12_a0d3,
        "resnet50_fc512_ms1_a0d1": resnet50_fc512_ms1_a0d1,
        "resnet50_fc512_ms123_a0d1": resnet50_fc512_ms123_a0d1,
        "resnet50_fc512_ms1234_a0d1": resnet50_fc512_ms1234_a0d1,
        "resnet50_fc512_ms23_a0d1": resnet50_fc512_ms23_a0d1,
        "resnet50_fc512_ms14_a0d1": resnet50_fc512_ms14_a0d1
    }

    backbone_name = cfg.backbone.name
    constructor = backbone_constructors.get(backbone_name)
    if constructor is None:
        raise ValueError(f"Backbone {backbone_name} not supported.")

    backbone = constructor(pretrained=pretrained, num_classes=cfg.data.num_classes)
    return backbone