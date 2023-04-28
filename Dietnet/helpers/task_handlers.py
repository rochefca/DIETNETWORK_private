import numpy as np
import h5py

import torch
import torch.nn.functional as F

class TaskHandler:
    def __init__(self, criterion, batches_results):
        self.criterion = criterion
        self.batches_results = batches_results
        self.best_results = dict()

    def format_ybatch(self, y_batch):
        return y_batch

    def compute_loss(self, model_out, y_batch):
        return self.criterion(model_out, y_batch)


class ClassificationHandler(TaskHandler):
    def __init__(self, dataset_filename, criterion):
        batches_results = {'losses':np.array([]),
                           'n_right':np.array([]), # nb samples with good pred
                           'preds':np.array([]), # the predicted class per sample
                           'scores':np.array([]), # samples softmax val per class
                           'ys':np.array([]), # labels of the batch
                           'samples':np.array([]) # samples of the batch
                         }

        super(ClassificationHandler, self).__init__(criterion, batches_results)

        # Label names
        f = h5py.File(dataset_filename, 'r')
        self.class_label_names = np.array(f['class_label_names'], dtype=np.str_)
        f.close()

        # Task name
        self.name = 'classification'

    def get_label(self, f, index):
        y = np.array(f['class_labels'][index], dtype=np.int64)

        return y

    def get_batch_results(self, model_out, y_batch):
        with torch.no_grad():
            # Scores : Softmax value per class for each sample
            scores = F.softmax(model_out, dim=1).detach().cpu().numpy()
            self.batches_results['scores'] = np.vstack(
                    [self.batches_results['scores'], scores]) \
                    if self.batches_results['scores'].size else scores

            # Pred : class prediction for each sample
            preds = np.argmax(scores, axis=1)
            self.batches_results['preds'] = np.concatenate(
                    [self.batches_results['preds'], preds])

            # Nb of samples for which the class was correctly predicted
            n_right_pred = ((y_batch.detach().cpu().numpy() - preds) == 0).sum()
            self.batches_results['n_right'] = np.append(
                    self.batches_results['n_right'], n_right_pred)


    def print_baseline_results(self, baseline_results):
        # Nb of samples
        nb_samples = len(baseline_results['samples'])

        # Baseline loss
        loss = baseline_results['losses'].sum()/nb_samples

        # Baseline accuracy
        acc = baseline_results['n_right'].sum()/float(nb_samples)*100

        print('Baseline loss: {} baseline acc: {}%'.format(loss, acc),
              flush=True)


    def print_epoch_results(self, train_results, valid_results):
        # Nb of samples
        nb_samples_train = len(train_results['samples'])
        nb_samples_valid = len(valid_results['samples'])

        # Loss
        loss_train = train_results['losses'].sum()/nb_samples_train
        loss_valid = valid_results['losses'].sum()/nb_samples_valid

        # Accuracy
        acc_train = train_results['n_right'].sum()/float(nb_samples_train)*100
        acc_valid = valid_results['n_right'].sum()/float(nb_samples_valid)*100

        print('train loss: {} train acc: {}%'
              '\nvalid loss: {} valid acc: {}%'.format(
              loss_train, acc_train, loss_valid, acc_valid), flush=True)


    def print_resumed_best_results(self, results):
        print('best valid loss: {} best valid acc: {}%'.format(
               results['loss'], results['acc']))


    def print_test_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])

        loss = results['losses'].sum()/nb_samples
        acc = results['n_right'].sum()/float(nb_samples)*100

        print('Test loss: {} test acc: {}%'.format(loss, acc))


    def init_best_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])
        # Best accuracy
        self.best_results['acc'] = results['n_right'].sum()/ \
                                   float(nb_samples)*100
        # Best loss
        self.best_results['loss'] = results['losses'].sum()/nb_samples


    def resume_best_results(self, results):
        self.best_results['acc'] = results['acc']
        self.best_results['loss'] = results['loss']


    def update_best_results(self, results):
        has_improved = False

        # Nb samples
        nb_samples = len(results['samples'])
        # Actual acc
        acc = results['n_right'].sum()/float(nb_samples)*100
        # Actual loss
        loss = results['losses'].sum()/nb_samples

        # Improvement if actual acc is greater than best acc
        if acc > self.best_results['acc']:
            has_improved = True
            # Update best acc
            self.best_results['acc'] = acc

        # Improvement if acc is same as best and loss is less than best
        if loss < self.best_results['loss'] and acc == self.best_results['acc']:
            has_improved = True
            # Update best loss
            self.best_results['loss'] = loss

        return has_improved

    def save_predictions(self, results, file_fullpath):
        np.savez(file_fullpath,
                 samples=results['samples'],
                 labels=results['ys'],
                 preds=results['preds'],
                 scores=results['scores'],
                 class_label_names=self.class_label_names)


class RegressionHandler(TaskHandler):
    def __init__(self, dataset_filename, criterion):
        # Task name
        self.name = 'regression'

        # Loss for this task
        self.criterion = criterion

        # Dict for saving results of each batch
        # (Will be { losses_wo_reduction:[...], preds:[...],
        # ys:[...], samples:[...] })
        self.batches_results = dict()

        # Dict for remembering results of the best epoch obtained so far
        # {mean_loss = float}
        self.best_epoch_results = dict()


    # RESHAPE : ADDING BATCH DIM
    def format_ybatch(self, y_batch):
        return y_batch.unsqueeze(1)


    # POUR AVOIR DE FLOAT 32?
    def get_label(self, f, index):
        y = np.array(f['regression_labels'][index], dtype=np.float32)

        return y


    def init_batches_results(self, dataset, dataloader):
        # Init the batch results with empty arrays
        self.batches_results['losses_wo_reduction'] = np.zeros(len(dataloader))
        self.batches_results['preds'] = np.zeros(len(dataset))
        self.batches_results['ys'] = np.zeros(len(dataset))
        self.batches_results['samples'] = np.zeros(len(dataset))


    def update_batches_preds(self, model_out, bstart, bend):
        # Model out is dim batch_size x 1 (each output is it's own tensor)
        # .squeeze() makes the dim batch_size (1 array of all outputs)
        self.batches_results['preds'][bstart:bend] = \
                model_out.detach().squeeze().cpu().numpy()


    def print_baseline_results(self, baseline_results):
        # Nb of samples
        nb_samples = len(baseline_results['samples'])

        # Baseline loss
        loss = baseline_results['losses_wo_reduction'].sum()/nb_samples

        print('Baseline loss: {}'.format(loss), flush=True)


    def print_epoch_results(self, train_results, valid_results):
        """
        Input
            train_results, valid resutls: a copy of
            self.batches_results = {losses_wo_reduction, preds, samples, ys}

        Output
            None, prints the mean train and valid epoch loss
        """

        # Nb of samples
        nb_samples_train = len(train_results['samples'])
        nb_samples_valid = len(valid_results['samples'])

        # Loss
        loss_train = train_results['losses_wo_reduction'].sum()/nb_samples_train
        loss_valid = valid_results['losses_wo_reduction'].sum()/nb_samples_valid

        print('train loss: {} valid loss: {}'.format(
              loss_train, loss_valid), flush=True)


    def print_resumed_best_results(self, results):
        print('best valid loss: {}'.format(results['mean_loss']))


    def print_test_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])

        loss = results['losses_wo_reduction'].sum()/nb_samples

        print('Test loss: {}'.format(loss))


    def init_best_epoch_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])
        # Best loss
        self.best_epoch_results['mean_loss'] = results['losses_wo_reduction'].sum()/nb_samples


    def resume_best_results(self, results):
        self.best_results['mean_loss'] = results['mean_loss']


    def update_best_results(self, results):
        has_improved = False

        # Nb samples
        nb_samples = len(results['samples'])
        # Actual loss
        loss = results['losses_wo_reduction'].sum()/nb_samples

        # Improvement if actual loss is less than best loss
        if loss < self.best_epoch_results['mean_loss']:
            has_improved = True
            # Update best loss
            self.best_epoch_results['mean_loss'] = loss

        return has_improved


    def save_predictions(self, results, file_fullpath):
        np.savez(file_fullpath,
                 samples=results['samples'],
                 labels=results['ys'],
                 preds=results['preds'])
