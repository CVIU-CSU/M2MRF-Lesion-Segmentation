from functools import reduce

import numpy as np
import torch
from mmcv.utils import print_log

from mmseg.core import metrics
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LesionDataset(CustomDataset):

    def __init__(self, **kwargs):
        super(LesionDataset, self).__init__(**kwargs)

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        # return super(LesionDataset, self).evaluate(results, metric, logger, **kwargs)
        return self._evaluate(results, metric, logger, **kwargs)

    def _evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU', 'mIoU2']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        iou, f1, ppv, s, aupr = metrics(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)  # evaluate
        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'F1', 'PPV', 'S', 'AUPR')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for i in range(num_classes):
            ppv_str = '{:.2f}'.format(ppv[i] * 100)
            s_str = '{:.2f}'.format(s[i] * 100)
            f1_str = '{:.2f}'.format(f1[i] * 100)
            iou_str = '{:.2f}'.format(iou[i] * 100)
            aupr_str = '{:.2f}'.format(aupr[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, f1_str, ppv_str, s_str, aupr_str)

        mIoU = np.nanmean(np.nan_to_num(iou[-4:], nan=0))
        mF1 = np.nanmean(np.nan_to_num(f1[-4:], nan=0))
        mPPV = np.nanmean(np.nan_to_num(ppv[-4:], nan=0))
        mS = np.nanmean(np.nan_to_num(s[-4:], nan=0))
        mAUPR = np.nanmean(np.nan_to_num(aupr[-4:], nan=0))

        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mF1', 'mPPV', 'mS', 'mAUPR')

        iou_str = '{:.2f}'.format(mIoU * 100)
        f1_str = '{:.2f}'.format(mF1 * 100)
        ppv_str = '{:.2f}'.format(mPPV * 100)
        s_str = '{:.2f}'.format(mS * 100)
        aupr_str = '{:.2f}'.format(mAUPR * 100)
        summary_str += line_format.format('global', iou_str, f1_str, ppv_str, s_str, aupr_str)

        eval_results['mIoU'] = mIoU
        eval_results['mF1'] = mF1
        eval_results['mPPV'] = mPPV
        eval_results['mS'] = mS
        eval_results['mAUPR'] = mAUPR

        # NEW: for two classes metric
        if metric == 'mIoU2':
            summary_str += '\n'

        print_log(summary_str, logger)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return eval_results
