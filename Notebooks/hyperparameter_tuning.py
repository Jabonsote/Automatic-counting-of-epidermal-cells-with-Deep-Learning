import cv2
import numpy as np
from scipy.cluster.vq import kmeans
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg

class AnchorCalculator:
    """
    A class for obtaining anchor sizes and ratios.
    """
    def __init__(self, trainer, cfg):
        """
        Initialize the AnchorCalculator with a trainer and a config.

        Args:
            trainer (detectron2.engine.defaults.DefaultTrainer): The trainer object responsible for data 
            loading and processing.
            
            cfg (detectron2.config.config.CfgNode): The configuration object containing model settings and 
            hyperparameters.
        """
        self.trainer = trainer
        self.cfg = cfg
        self.iterator = 0

        # Initialize data loader and get the first data batch
        trainer._data_loader_iter = iter(self.trainer.data_loader)
        self.data = next(trainer._data_loader_iter)
        self.generate_anchors = self.trainer.model.proposal_generator.anchor_generator.generate_cell_anchors

    def visualize_images_gt_box(self):
        """
        Preprocess and visualize the images from the data batch.
        """
        images = self.trainer.model.preprocess_image(self.data)
        pixel_mean = self.cfg.MODEL.PIXEL_MEAN
        pixel_std = self.cfg.MODEL.PIXEL_STD

        for i in range(len(images)):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * pixel_std) + pixel_mean

            # Visualize the image with ground-truth boxes
            v = Visualizer(img, metadata={}, scale=0.5)
            v = v.overlay_instances(boxes=self.data[i]["instances"].get("gt_boxes").tensor)
            
            dpi = 80
            im_data = v.get_image()[:, :, ::-1]
            height, width, depth = im_data.shape
            figsize = width / float(dpi), height / float(dpi)
            fig = plt.figure(figsize=figsize)
            plt.imshow(im_data)
            plt.axis("off")
            plt.show()
        

    def get_gt_boxes_batch(self):
        """
        Get ground-truth boxes from a batch of data.
        """
        gt_boxes = [item['instances'].get('gt_boxes').tensor for item in self.data]
        return torch.cat(gt_boxes)

    def get_gt_boxes(self, iterations, verbose = True):
        """
        Get ground-truth boxes from the training dataset.
        """
        pbar = tqdm(range(iterations)) if verbose else range(iterations)
              
        gt_boxes = [self.get_gt_boxes_batch() for _ in pbar]
        self.iterator = iterations
        return torch.cat(gt_boxes)

    def boxes2wh(self, boxes = None, verbose = False):
        """
        Convert bounding boxes to width and height (wh) format.
        """
        if boxes == None:
            boxes = self.get_gt_boxes(self.iterator, verbose)
            x1y1 = boxes[:, :2]
            x2y2 = boxes[:, 2:]

        else:
            x1y1 = boxes[:, :2]
            x2y2 = boxes[:, 2:]
        
        return x2y2 - x1y1

    
    def wh2size(self):
        """
        Calculate anchor sizes from width and height (wh).
        """
        wh = self.boxes2wh()
        return torch.sqrt(wh[:, 0] * wh[:, 1])

    
    def wh2ratio(self):
        """
        Calculate anchor ratios from width and height (wh).
        """
        wh = self.boxes2wh()
        return wh[:, 1] / wh[:, 0]

    
    def best_ratio(self, anchors = None):
        """
        Calculate the best anchor ratios based on ground-truth boxes.
        """
        if anchors == None:
            anchors = self.generate_anchors()
            ac_wh = self.boxes2wh(boxes = anchors)

        else:
            ac_wh = self.boxes2wh(boxes = anchors)
            
            
        gt_wh = self.boxes2wh()
        
        all_ratios = gt_wh[:, None] / ac_wh[None]
        inverse_ratios = 1 / all_ratios
        ratios = torch.min(all_ratios, inverse_ratios)
        worst = ratios.min(-1).values
        best = worst.max(-1).values
        return best

    
    def fitness(self, anchors = None,  EDGE_RATIO_THRESHOLD=0.25):
        """
        Compute the fitness of anchor sizes and ratios.
        """
        ratio = self.best_ratio(anchors)
        return (ratio * (ratio > EDGE_RATIO_THRESHOLD).float()).mean()

    
    def best_recall(self, EDGE_RATIO_THRESHOLD=0.25):
        """
        Calculate the best recall of anchor sizes and ratios based on ground-truth boxes.
        """
        ratio = self.best_ratio()
        best = (ratio > EDGE_RATIO_THRESHOLD).float().mean()
        return best


class anchorSizeRatioTuner:
    """
    A class for tuning anchor sizes and ratios using clustering and genetic algorithms.
    """
    
    def __init__(self, trainer, fit_fn):
        """
        Initialize the HyperparameterTuner with anchor sizes, ratios, ground-truth box sizes,
        anchor generation function, and fitness function.

        Args:
            trainer (detectron2.engine.defaults.DefaultTrainer): The trainer object responsible for data 
            loading and processing.
            
            fit_fn (function): Fitness function to quantify the quality of anchor sizes and ratios.
        """
        self.trainer = trainer
        
        self.generate_anchors = self.trainer.model.proposal_generator.anchor_generator.generate_cell_anchors
        self.fit_fn = fit_fn

    def estimate_clusters(self, values, num_clusters, iter=100):
        """
        Estimate anchor sizes and ratios using clustering.

        Args:
            values (torch.Tensor): Values to be clustered.
            num_clusters (int): Number of clusters to estimate.
            iter (int): Number of iterations for clustering.

        Returns:
            torch.Tensor: Estimated cluster centers for anchor sizes and ratios.
        """
        std = values.std(0).item()
        k, _ = kmeans(values / std, num_clusters, iter=iter)
        k *= std
        return k
        
    def visualize_clusters(self, values, centers):
        """
        Visualize anchor size and ratio clusters.

        Args:
            values (torch.Tensor): Values used for clustering.
            centers (torch.Tensor): Cluster centers.
        """
        plt.hist(values, histtype='step')
        plt.scatter(centers, [0] * len(centers), c="red")
        plt.show()

    def evolve(self, sizes, ratios, iterations=10000, probability=0.9, muy=1, sigma=0.05, verbose=False):
        """
        Evolve anchor sizes and ratios using a genetic algorithm.

        Args:
            sizes (list): Initial anchor sizes.
            ratios (list): Initial anchor ratios.
            iterations (int): Number of iterations for the genetic algorithm.
            probability (float): Probability of mutation.
            muy (float): Mean of the mutation.
            sigma (float): Standard deviation of the mutation.
            verbose (bool): Whether to display progress.

        Returns:
            list: Tuned anchor sizes and ratios.
        """
        # Step 1: Current anchors and their fitness
        anchors = self.generate_anchors(tuple(sizes), tuple(ratios))
        best_fit = self.fit_fn(anchors = anchors)
        anchor_shape = len(sizes) + len(ratios)

        pbar = tqdm(range(iterations), desc=f"Evolving ratios and sizes:")
        for i, _ in enumerate(pbar):
            # to mutate and how much
            mutation = np.ones(anchor_shape)
            mutate = np.random.random(anchor_shape) < probability
            mutation = np.random.normal(muy, sigma, anchor_shape) * mutate
            mutation = mutation.clip(0.3, 3.0)
            # mutated
            mutated_sizes = sizes.copy() * mutation[:len(sizes)]
            mutated_ratios = ratios.copy() * mutation[-len(ratios):]
            mutated_anchors = self.generate_anchors(tuple(mutated_sizes), tuple(mutated_ratios))
            mutated_fit = self.fit_fn(anchors = mutated_anchors)

            if mutated_fit > best_fit:
                sizes = mutated_sizes.copy()
                ratios = mutated_ratios.copy()
                best_fit = mutated_fit
                pbar.desc = (f"Evolving ratios and sizes, Fitness = {best_fit:.4f}")

        return sizes, ratios
        
class ImagesMeanStd:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.ssd = 0

    def broad_cast(self, x, image_size, channel):
        y = torch.broadcast_to(x, (image_size**2, channel))
        z = y.reshape(image_size, image_size, 3)
        return torch.moveaxis(z, 2, 0)
    
    def push(self, x):
        # start
        dims = [0, 2, 3]
        count = 1
        for dim in dims:
            count *= x.shape[dim]
        image_size = x.shape[-1]
        channel = x.shape[1]
        if self.n == 0:
            # start
            new_mean = x.sum(axis=dims)/count
            new_ssd = ((
                x - self.broad_cast(
                    new_mean, 
                    image_size, 
                    channel
                ))**2).sum(axis=dims)
            new_count = count
        else:
            # old
            old_count = self.n
            old_mean = self.mean
            old_ssd = self.ssd
            old_sum = old_mean * old_count      
            # new
            new_count = self.n + count
            new_sum = old_sum + x.sum(axis=dims)
            new_mean = new_sum/(self.n + count)
      
            old_ssd_new_mean = (
                old_ssd  
                + 2*old_mean*old_sum
                - old_count*(old_mean)**2
                - 2*new_mean*old_sum
                + old_count*(new_mean)**2
                )
      
            new_ssd = (
                old_ssd_new_mean + 
                (
                    (x - self.broad_cast(new_mean, 
                                image_size, 
                                channel))**2
                ).sum(axis=dims))
        # release results
        self.mean = new_mean
        self.ssd = new_ssd
        self.n = new_count
        self.std = torch.sqrt(new_ssd/(new_count-1))








    