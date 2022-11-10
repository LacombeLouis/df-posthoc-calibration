import numpy as np
from calibration import HB_binary


class top_label_calibration(object):
    def __init__(self, points_per_bin=50):
        ### Hyperparameters
        self.points_per_bin = points_per_bin
    
    def fit(self, preds_calib, y_calib):
        top_class_prob = np.max(preds_calib, axis=1)
        top_class_prob_arg = np.argmax(preds_calib, axis=1)+1

        calibrators = {}
        for item in np.unique(top_class_prob_arg):
            correct_label = np.where(top_class_prob_arg == item)[0]
            n_l = len(correct_label)
            bins_l = np.floor(n_l/self.points_per_bin).astype('int')
            if(bins_l == 0):
                print("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}.".format(item, points_per_bin, item))
            else:
                hb_clone = HB_binary(n_bins=bins_l)
                y_calib_ = (y_calib[correct_label] == item)
                top_class_prob_ = top_class_prob[correct_label]
                hb_clone.fit(top_class_prob_, y_calib_)
                calibrators[item] = hb_clone

        self.calibrators = calibrators

    def predict_proba(self, pred_test):
        calibrators = self.calibrators
        top_class_prob_TEST = np.max(pred_test, axis=1)
        top_class_prob_arg_TEST = np.argmax(pred_test, axis=1)+1

        n = len(top_class_prob_TEST)
        calibrated_test_values = np.zeros((n))
        for item in np.unique(top_class_prob_arg_TEST):
            if item in calibrators:
                correct_label = np.where(top_class_prob_arg_TEST == item)[0]
                if len(correct_label) > 0:
                    calibrator_ = calibrators[item]
                    preds_ = calibrator_.predict_proba(top_class_prob_TEST[correct_label])
                    calibrated_test_values[correct_label] = preds_

        return calibrated_test_values
