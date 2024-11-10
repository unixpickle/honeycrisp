# ``Tensor/scatter(axis:count:indices:indicesAreUnique:function:file:line:)``

Scatter values from the source ``Tensor`` along the specified axis at the given indices. The caller must specify the new size of the given axis, and no indices should exceed this size. If the indices are unique (per slice of indices), then you may pass `indicesAreUnique: true` and possibly achieve a faster computation.