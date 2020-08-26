import os
import shutil
import warnings
import numpy as np
import rasterio.transform
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import torch
import torch.optim
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import torch.utils.tensorboard

# to ensure reproducibility
torch.manual_seed(0)


def get_distribution(tensor, k=1):
    """Generates a dict of informative summary stats for tensor.

    Args:
        tensor (torch.Tensor [N,]): tensor of interest
        k (int): k for top/bottom k, the largest/smallest k values will be
            returned

    Returns:
        dict: {str: float}
    """
    top, _ = torch.topk(tensor, k=k, largest=True)
    bottom, _ = torch.topk(tensor, k=k, largest=False)
    output = {'mean': tensor.mean()}
    output.update(
        {'top{}'.format(i + 1): v.item()
         for i, v in enumerate(top)})
    output.update(
        {'bottom{}'.format(i + 1): v.item()
         for i, v in enumerate(bottom)})
    return output


def optimize(nodes, links, width, height,
             thetas_init, scales_init, xs_init, ys_init,
             n_iter=1, lr_theta=0, lr_scale=0, lr_xy=0,
             lr_scheduler_milestones=None,
             output_iter=None, logging=False, logdir=None, verbose=True):
    """Globally minimize loss by adjusting affine transforms.

    The loss function is the mean of distance between true relative
    affine transformations between pairs of images, as estimated in the
    stitching procedures, and estimated relative affine transformations,
    as calculated from parameterized affine transformations that are
    optimized over.

    Args:
        nodes (list of (int, int)): ids of images in the graph
        links (dict): {((int, int), (int, int)): affine.Affine} "true"
            transforms between pairs of images that are estimated in the
            stitching procedures
        width, height (list of int [len(nodes),]): (width, height) of images
        thetas_init (list or numpy.ndarray of float [len(nodes),]): initial
            values for rotation parameters (in world crs), in radian,
            clockwise (ccw rotation matrix is used but produces cw rotation
            in non standard coord space (y axis points downward))
        scales_init (list or numpy.ndarray of float [len(nodes),]): initial
            values for scale parameters (in world crs)
        xs_init, ys_init (list or numpy.ndarray of float [len(nodes),]):
            initial values for x and y translation shifts (in world crs)
            these refer to the centroids
        n_iter (int): number of iterations
        lr_theta, lr_scale, lr_xy (float): param specific learning rates
            for the adam optimizer
        lr_scheduler_milestones (NoneType or list of int): iterations when
            learning rate is decayed by 0.1, disabled if None
        output_iter (list of int): iterations where loss and affines are
            returned as outputs, if None, this defaults to [last iteration]
        logging (bool): whether to turn on TensorBoard logging
        logdir (str): where to store tf events files
        verbose (bool)

    Returns:
        tuple (list of int, list of float, list of list of affine.Affine):
            iterations, losses at those iterations,
            transforms estimated at those iterations
    """
    if logging:
        if os.path.isdir(logdir):
            shutil.rmtree(logdir)
        os.makedirs(logdir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=logdir)
        # init graph
        fig, ax = plt.subplots(figsize=(15, 6))
        # invert y axis
        ax.invert_yaxis()
        # iterate over images
        for x, y, scale, theta, w, h, node in zip(
                xs_init, ys_init, scales_init, thetas_init,
                width, height, nodes):
            # plot centroids
            ax.plot(x, y, 'k+')
            # plot rotation & scale
            ax.arrow(
                x, y,
                h / 2 * scale * np.sin(theta),
                - h / 2 * scale * np.cos(theta),
                head_width=5, head_length=10,
                fc='darkseagreen', ec='darkseagreen')
            # annotate image ids
            ax.annotate(
                s='{}-{}'.format(*node), xy=(x, y), fontsize=7)
        # iterate over links
        for (i, j), trans in links.items():
            # plot links
            ax.plot([xs_init[nodes.index(i)], xs_init[nodes.index(j)]],
                    [ys_init[nodes.index(i)], ys_init[nodes.index(j)]],
                    'grey' if trans is None else 'hotpink')
        writer.add_figure('graph_init', fig, global_step=0)

    if output_iter is None:
        # output results from last iteration
        output_iter = [n_iter - 1]

    # prepare for init
    rel_true_tensor = []  # collects true relative links
    affines_i_idx = []  # collects integer indices for i in link(i, j)
    affines_j_idx = []  # collects integer indices for j in link(i, j)
    pts_tensor = []  # collects points (corners)
    for (i_idx, j_idx), trans in links.items():
        if trans is not None:
            rel_true_tensor.append(torch.cat(
                [torch.tensor(trans.column_vectors).T,
                 torch.tensor([[0., 0., 1.]])], dim=0))
            affines_i_idx.append(nodes.index(i_idx))
            affines_j_idx.append(nodes.index(j_idx))
            j_width = width[nodes.index(j_idx)]
            j_height = height[nodes.index(j_idx)]
            # points: four corners of the j image
            pts_tensor.append(torch.tensor([
                [0, 0, 1],
                [0, j_height, 1],
                [j_width, 0, 1],
                [j_width, j_height, 1]], dtype=torch.float).T)
    if len(rel_true_tensor) == 0:
        raise ValueError('No links available for optimization.')
    rel_true_tensor = torch.stack(rel_true_tensor)  # [n_links, 3, 3]
    pts_tensor = torch.stack(pts_tensor)  # [n_links, 3, n_pts]

    # convert to tensor
    width = torch.tensor(width, dtype=torch.float)
    height = torch.tensor(height, dtype=torch.float)
    # initialize leaf nodes
    thetas = torch.tensor(thetas_init, dtype=torch.float, requires_grad=True)
    scales = torch.tensor(scales_init, dtype=torch.float, requires_grad=True)
    xs = torch.tensor(xs_init, dtype=torch.float, requires_grad=True)
    ys = torch.tensor(ys_init, dtype=torch.float, requires_grad=True)

    # initialize optimizer and scheduler
    optimizer_scale = torch.optim.Adam([scales], lr=lr_scale)
    optimizer_theta = torch.optim.Adam([thetas], lr=lr_theta)
    optimizer_xy = torch.optim.Adam([xs, ys], lr=lr_xy)
    if lr_scheduler_milestones is not None:
        lr_scheduler_scale = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_scale, lr_scheduler_milestones)
        lr_scheduler_theta = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_theta, lr_scheduler_milestones)
        lr_scheduler_xy = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_xy, lr_scheduler_milestones)

    # prepare to collect output
    assert (n_iter - 1) in output_iter, 'Last iteration not saved!'
    output_loss = []
    output_affines = []

    # iterate n_iter times
    for k in range(n_iter):
        optimizer_scale.zero_grad()
        optimizer_theta.zero_grad()
        optimizer_xy.zero_grad()
        # compute absolute affine transformation
        affines = torch.stack([  # [n_nodes, 3, 3]
            torch.cos(thetas) * scales,
            - torch.sin(thetas) * scales,
            xs - width / 2 * scales * torch.cos(thetas) +
            height / 2 * scales * torch.sin(thetas),
            torch.sin(thetas) * scales,
            torch.cos(thetas) * scales,
            ys - width / 2 * scales * torch.sin(thetas) -
            height / 2 * scales * torch.cos(thetas),
            torch.zeros(len(nodes), dtype=torch.float),
            torch.zeros(len(nodes), dtype=torch.float),
            torch.ones(len(nodes), dtype=torch.float)]).T.view(-1, 3, 3)
        # extract i, j affines to estimate relative affines
        affines_i = affines[affines_i_idx, ...]  # [n_links, 3, 3]
        affines_j = affines[affines_j_idx, ...]  # [n_links, 3, 3]
        # get estimated relative affines
        rel_est_tensor = torch.matmul(
            torch.inverse(affines_i), affines_j)  # [n_links, 3, 3]
        # compute loss on each link (between true/estimated relative affines)
        # loss is the mean squared distance between points
        # in the true versus estimated relative affines
        losses = (  # [n_links, 3, n_pts]
            (torch.matmul(rel_est_tensor, pts_tensor) -
             torch.matmul(rel_true_tensor, pts_tensor)) ** 2
        ).sum(axis=1).mean(axis=1)  # -> [n_links, n_pts] -> [n_links,]
        loss = losses.mean()  # -> [1,]
        if logging:
            if k % 500 == 0:
                writer.add_scalar('loss/mean', loss.item(), k)
                writer.add_scalars(
                    'affines/thetas', get_distribution(thetas), k)
                writer.add_scalars(
                    'affines/scales', get_distribution(scales), k)
                writer.add_scalars(
                    'affines/xs', get_distribution(xs), k)
                writer.add_scalars(
                    'affines/ys', get_distribution(ys), k)
                # link loss graph
                # color palette
                cmap = matplotlib.cm.get_cmap('viridis_r')
                norm = matplotlib.colors.Normalize(
                    vmin=0, vmax=torch.max(losses).item(), clip=True)
                fig, ax = plt.subplots(figsize=(15, 6))
                # invert y axis
                ax.invert_yaxis()
                for x, y, node in zip(
                        xs.detach().numpy(), ys.detach().numpy(), nodes):
                    # plot centroids
                    ax.scatter(x, y, color='dimgray', marker='+')
                    # annotate image ids
                    ax.annotate(
                        s='{}-{}'.format(*node), xy=(x, y), fontsize=7)
                # iterate over links
                for link_idx, link_loss in enumerate(losses):
                    # plot links, color corresponds to loss on link
                    ax.plot(
                        [xs[affines_i_idx[link_idx]].item(),
                         xs[affines_j_idx[link_idx]].item()],
                        [ys[affines_i_idx[link_idx]].item(),
                         ys[affines_j_idx[link_idx]].item()],
                        color=cmap(norm(link_loss.item())))
                fig.colorbar(
                    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
                writer.add_figure(
                    'link_loss_init' if k == 0 else 'link_loss',
                    fig, global_step=k)
                # print links of largest loss
                link_losses, link_idxs = torch.topk(losses, k=5, largest=True)
                for i, (link_idx, link_loss) in enumerate(zip(
                        link_idxs, link_losses)):
                    writer.add_text(
                        'loss_topk{}/{}'.format('_init' if k == 0 else '', i),
                        'Loss: {} between {}-{} and {}-{}'.format(
                            link_loss,
                            *nodes[affines_i_idx[link_idx]],
                            *nodes[affines_j_idx[link_idx]]),
                        global_step=k)
        if verbose:
            if (k + 1) % 200 == 0:
                print('Iter: {}; Loss: {:.3f}'.format(k, loss.item()))
        # output affines and loss
        if k in output_iter:
            output_loss.append(loss.item())
            output_affine = [
                rasterio.transform.Affine(*img_affine[0:2, :].flatten())
                for img_affine in affines.detach().numpy()]
            output_affines.append(output_affine)
        # back propagate
        loss.backward()
        optimizer_theta.step()
        optimizer_scale.step()
        optimizer_xy.step()
        if lr_scheduler_milestones is not None:
            # learning rate update
            lr_scheduler_theta.step()
            lr_scheduler_scale.step()
            lr_scheduler_xy.step()

    if logging:
        writer.close()

    return output_iter, output_loss, output_affines
