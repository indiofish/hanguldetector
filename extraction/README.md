# Feature Extraction from a given font image

This is a very crude algorithm, that kinda works.

Drawbacks:
The composition of border boxes of a same character can differ,
if the font is different.


One possible solution would be to extract the outermost border.
However, this means that we can only get 4 values(if we normalize) per
character, and this is not enough to distinguish characters.

One other solution would be to use the outermost border, but use a different
normalizing strategy, that gives more than 4 numbers.
