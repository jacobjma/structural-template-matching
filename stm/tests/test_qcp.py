import numpy as np
from stm.segment import segmentation
from stm.register import lattice

for i in range(10):
    a=np.random.rand(2)
    b=np.random.rand(2)
    n=np.random.randint(3,10)

    t = lattice.create_template(a, b, n_points=n)
    
    points = t + np.random.randn(t.shape[0],t.shape[1])

    S = segmentation.segment_neighbors(points,n)

    S.match(t,rmsd_algorithm='qcp')

    rmsd_qcp = S.rmsd

    S.match(t,rmsd_algorithm='kabsch')

    rmsd_kabsch = S.rmsd

    assert all(np.isclose(rmsd_qcp, rmsd_kabsch))