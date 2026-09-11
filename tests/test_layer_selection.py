from types import SimpleNamespace

from networkgeometry.run import select_plot_layers


def _circularity(angular_order_by_layer):
    return [SimpleNamespace(layer=layer, angular_order=value)
            for layer, value in angular_order_by_layer.items()]


def test_selects_layer_high_on_both_metrics_not_either_alone():
    # Layer 2 is strong on circularity AND within-AUC; layer 0 is high only on within-AUC
    # (the trivial early-layer case), layer 4 only on circularity. Only layer 2 wins.
    part1 = {"day": _circularity({0: 0.10, 2: 0.90, 4: 0.95}),
             "month": _circularity({0: 0.20, 2: 0.85, 4: 0.90})}
    within = {"day": {0: 0.99, 2: 0.95, 4: 0.50},
              "month": {0: 0.98, 2: 0.90, 4: 0.55}}
    assert select_plot_layers(part1, within, k=1) == [2]


def test_returns_k_layers_in_ascending_order():
    part1 = {"day": _circularity({0: 0.1, 1: 0.5, 2: 0.9, 3: 0.7})}
    within = {"day": {0: 0.9, 1: 0.5, 2: 0.8, 3: 0.95}}
    layers = select_plot_layers(part1, within, k=2, names=("day",))
    assert len(layers) == 2
    assert layers == sorted(layers)


def test_only_uses_structures_present_in_both_inputs():
    part1 = {"day": _circularity({0: 0.1, 1: 0.9}),
             "month": _circularity({0: 0.2, 1: 0.8})}
    within = {"day": {0: 0.5, 1: 0.95}}  # month missing -> ignored, no KeyError
    assert select_plot_layers(part1, within, k=1) == [1]
