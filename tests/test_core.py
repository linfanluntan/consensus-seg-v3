import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.consensus_methods import majority_vote, staple_binary, spatial_staple_binary, simple_fusion, hybrid_fusion, log_opinion_pool, staple_multiclass
from src.metrics import dice, make_phantom, make_raters, make_multiclass_phantom, make_multiclass_raters, ece

def test_mv(): assert np.all(majority_vote(np.ones((3,10,10),dtype=int)) == 1)
def test_staple_perfect():
    gt, _ = make_phantom((64,64)); R = np.stack([gt]*3)
    assert dice(staple_binary(R)['consensus'], gt) > 0.99
def test_staple_convergence():
    gt, _ = make_phantom((64,64)); R, _, _ = make_raters(gt, 5)
    ll = staple_binary(R, max_iter=50)['log_likelihood']
    for i in range(1,len(ll)): assert ll[i] >= ll[i-1]-1e-3
def test_spatial():
    gt, _ = make_phantom((64,64)); R, _, _ = make_raters(gt, 5)
    assert dice(spatial_staple_binary(R)['consensus'], gt) > 0.5
def test_simple():
    gt, img = make_phantom((64,64)); R, _, _ = make_raters(gt, 5)
    assert dice(simple_fusion(R, image=img)['consensus'], gt) > 0.7
def test_hybrid():
    gt, img = make_phantom((64,64)); R, _, _ = make_raters(gt, 5)
    assert dice(hybrid_fusion(R, img)['consensus'], gt) > 0.7
def test_dice(): assert dice(np.ones((10,10),dtype=int), np.ones((10,10),dtype=int)) > 0.999
def test_ece_val(): assert ece(np.array([1.,0,1,0]), np.array([1,0,1,0]))[0] < 0.05
def test_multiclass():
    gt = make_multiclass_phantom((64,64), K=3); R, _ = make_multiclass_raters(gt, 3, 5)
    assert staple_multiclass(R, 3)['consensus'].shape == gt.shape
def test_log_pool():
    gt, _ = make_phantom((64,64)); R, _, _ = make_raters(gt, 5)
    assert dice(log_opinion_pool(R)['consensus'], gt) > 0.5
def test_exclude_consensus():
    gt, _ = make_phantom((64,64)); R, _, _ = make_raters(gt, 5)
    assert dice(staple_binary(R, exclude_consensus=True)['consensus'], gt) > 0.5
