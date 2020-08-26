import pytest

import numpy as np
import torch
import rasterio
import rasterio.transform

from ..optim import get_distribution, optimize


@pytest.mark.parametrize(
    'tensor,k,expected',
    [
        pytest.param(
            torch.tensor([2, 1, 9, 8], dtype=torch.float, requires_grad=True),
            2,
            {'mean': 5.,
             'top1': 9., 'top2': 8.,
             'bottom1': 1., 'bottom2': 2.},
        ),
    ]
)
def test_get_distribution(tensor, k, expected):
    assert get_distribution(tensor=tensor, k=k) == expected


@pytest.mark.parametrize(
    'expected_loss,expected_affines,'
    'nodes,links,width,height,'
    'thetas_init,scales_init,xs_init,ys_init,'
    'n_iter,lr_theta,lr_scale,lr_xy,'
    'lr_scheduler_milestones,output_iter',
    [
        # case 1: correct init, loss = 0, no rotation or scaling
        pytest.param(
            # expected_loss
            [0],
            # expected_affines
            [[rasterio.transform.Affine.translation(50, 60),
              rasterio.transform.Affine.translation(60, 60),
              rasterio.transform.Affine.translation(75, 60)]],
            # nodes
            [(12, 4), (15, 3), (56, 1)],
            # links
            {
                ((12, 4), (15, 3)):
                rasterio.transform.Affine.translation(10, 0),
                ((15, 3), (56, 1)):
                rasterio.transform.Affine.translation(15, 0),
                ((12, 4), (56, 1)): None,
            },
            # width
            [20, 20, 20],
            # height
            [30, 30, 30],
            # thetas_init
            [0, 0, 0],
            # scales_init
            [1, 1, 1],
            # xs_init
            [60, 70, 85],
            # ys_init
            [75, 75, 75],
            # n_iter
            1,
            # lr_theta, lr_scale, lr_xy
            0, 0, 0,
            # lr_scheduler_milestones
            None,
            # output_iter
            None,
        ),
        # case 2: init, check loss = dist, no rotation or scaling
        pytest.param(
            # expected_loss
            [25 / 2],
            # expected_affines
            [[rasterio.transform.Affine.translation(50, 60),
              rasterio.transform.Affine.translation(60, 60),
              rasterio.transform.Affine.translation(79, 57)]],
            # nodes
            [(12, 4), (15, 3), (56, 1)],
            # links
            {
                ((12, 4), (15, 3)):
                rasterio.transform.Affine.translation(10, 0),
                ((15, 3), (56, 1)):
                rasterio.transform.Affine.translation(15, 0),
                ((12, 4), (56, 1)): None,
            },
            # width
            [20, 20, 20],
            # height
            [30, 30, 30],
            # thetas_init
            [0, 0, 0],
            # scales_init
            [1, 1, 1],
            # xs_init
            [60, 70, 89],
            # ys_init
            [75, 75, 72],
            # n_iter
            1,
            # lr_theta, lr_scale, lr_xy
            0, 0, 0,
            # lr_scheduler_milestones
            None,
            # output_iter
            None,
        ),
        # case 3: init, check loss = dist, + rotation, + scaling
        pytest.param(
            # expected_loss
            [(np.array([
                [10 - 30, 0 - 40],
                [10 - 110, 40 - 40],
                [30 - 30, 0 - 0],
                [30 - 110, 40 - 0],
            ]) ** 2).sum(axis=1).mean()],
            # expected_affines
            [[rasterio.transform.Affine.translation(80, 35) *
              # clockwise, in degrees
              rasterio.transform.Affine.rotation(90) *
              rasterio.transform.Affine.scale(0.5),
              rasterio.transform.Affine.translation(60, 50),
              rasterio.transform.Affine.translation(70, 50) *
              # clock wise, in degrees
              rasterio.transform.Affine.rotation(-90) *
              rasterio.transform.Affine.scale(2)]],
            # nodes
            [(12, 4), (15, 3), (56, 1)],
            # links
            {
                ((12, 4), (15, 3)):
                rasterio.transform.Affine.translation(10, 0),
                ((15, 3), (56, 1)): None,
                ((12, 4), (56, 1)): None,
            },
            # width
            [20, 20, 20],
            # height
            [40, 40, 40],
            # thetas_init, clockwise
            [np.pi / 2, 0, - np.pi / 2],
            # scales_init
            [0.5, 1, 2],
            # xs_init
            [70, 70, 110],
            # ys_init
            [40, 70, 30],
            # n_iter
            1,
            # lr_theta, lr_scale, lr_xy
            0, 0, 0,
            # lr_scheduler_milestones
            None,
            # output_iter
            None,
        ),
        # case 4: gradient descent, convergence w/ no conflict
        pytest.param(
            # expected_loss
            [8, 0],
            # expected_affines
            [[rasterio.transform.Affine.identity(),
              rasterio.transform.Affine.translation(2, -2),
              # orphaned image does not move
              rasterio.transform.Affine.translation(10, 30)],
             # iteration 99
             [rasterio.transform.Affine.translation(1, -1),
              rasterio.transform.Affine.translation(1, -1),
              # orphaned image does not move
              rasterio.transform.Affine.translation(10, 30)]],
            # nodes
            [(12, 4), (15, 3), (56, 1)],
            # links
            {
                ((12, 4), (15, 3)):
                rasterio.transform.Affine.identity(),
                ((15, 3), (12, 4)):
                rasterio.transform.Affine.identity(),
                ((15, 3), (56, 1)): None,
                ((12, 4), (56, 1)): None,
            },
            # width
            [2, 2, 2],
            # height
            [2, 2, 2],
            # thetas_init, clockwise
            [0, 0, 0],
            # scales_init
            [1, 1, 1],
            # xs_init
            [1, 3, 11],
            # ys_init
            [1, -1, 31],
            # n_iter
            100,
            # lr_theta, lr_scale, lr_xy
            0.00001, 0.00001, 0.15,
            # lr_scheduler_milestones
            [60],
            # output_iter
            [0, 99],
        ),
        # case 5: gradient descent, convergence w/ conflict
        pytest.param(
            # expected_loss
            [25 / 2, 3.6],
            # expected_affines
            [[rasterio.transform.Affine.identity(),
              rasterio.transform.Affine.identity(),
              rasterio.transform.Affine.translation(0, 4)],
             # iteration 99
             [rasterio.transform.Affine.translation(0.6, 2),
              rasterio.transform.Affine.translation(-0.6, 2),
              rasterio.transform.Affine.translation(0, 2)]],
            # nodes
            [(12, 4), (15, 3), (56, 1)],
            # links
            {
                ((12, 4), (15, 3)):
                rasterio.transform.Affine.identity(),
                ((15, 3), (12, 4)):
                rasterio.transform.Affine.identity(),
                ((15, 3), (56, 1)):
                rasterio.transform.Affine.translation(3, 0),
                ((12, 4), (56, 1)):
                rasterio.transform.Affine.translation(-3, 0),
            },
            # width
            [2, 2, 2],
            # height
            [2, 2, 2],
            # thetas_init, clockwise
            [0, 0, 0],
            # scales_init
            [1, 1, 1],
            # xs_init
            [1, 1, 1],
            # ys_init
            [1, 1, 5],
            # n_iter
            100,
            # lr_theta, lr_scale, lr_xy
            0.00001, 0.00001, 0.15,
            # lr_scheduler_milestones
            [60],
            # output_iter
            [0, 99],
        ),
    ],
)
def test_optimize(expected_loss, expected_affines,
                  nodes, links, width, height,
                  thetas_init, scales_init, xs_init, ys_init,
                  n_iter, lr_theta, lr_scale, lr_xy,
                  lr_scheduler_milestones,
                  output_iter):
    output_iter, output_loss, output_affines = optimize(
        nodes=nodes, links=links, width=width, height=height,
        thetas_init=thetas_init, scales_init=scales_init,
        xs_init=xs_init, ys_init=ys_init,
        n_iter=n_iter, lr_theta=lr_theta, lr_scale=lr_scale, lr_xy=lr_xy,
        lr_scheduler_milestones=lr_scheduler_milestones,
        output_iter=output_iter)
    assert output_loss == pytest.approx(expected_loss, rel=1e-4, abs=1e-4)
    # iterate over iterations
    for output_affines_iter, expected_affines_iter in zip(
            output_affines, expected_affines):
        # iterate over nodes
        for output_affine, expected_affine in zip(
                output_affines_iter, expected_affines_iter):
            assert output_affine == pytest.approx(
                expected_affine, rel=1e-4, abs=1e-2)
