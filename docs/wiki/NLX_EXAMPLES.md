# Grounded NLX examples

The orange evidence tiles are shown in [activation_tree_grounded_examples.png](assets/activation_tree_grounded_examples.png). Each sentence below uses only a verified tree condition and either an exact tile or visible CUB part points inside that tile.

## Sample 2 — predicted class 0

> The activation tree predicts class 0 because activation channel 242 in spatial tile (row 4, column 3) is ≤ −0.314; and activation channel 68 in the back, breast, left wing, tail area is ≤ 8.764.

## Sample 62 — predicted class 1

> The activation tree predicts class 1 because activation channel 242 in the left leg area is > −0.314; and activation channel 12 in the nape area is ≤ −2.254.

## Sample 121 — predicted class 2

> The activation tree predicts class 2 because activation channel 242 in spatial tile (row 4, column 3) is ≤ −0.314; activation channel 68 in the beak, crown, forehead, left eye, nape, right eye, throat area is > 8.764; and activation channel 101 in the back area is > −0.129.

## Sample 181 — predicted class 3

> The activation tree predicts class 3 because activation channel 242 in spatial tile (row 4, column 3) is > −0.314; activation channel 12 in the crown, forehead, right eye area is > −2.254; activation channel 321 in spatial tile (row 3, column 3) is > 2.073; and activation channel 127 in the belly, left leg area is ≤ −1.753.

## Scope

These are faithful template NLXs, not attribute-level descriptions. Part names are used only where CUB’s visible part-point annotation lies in the certified tile.
