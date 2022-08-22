import numpy as np
import h5py

import torch
import torch.nn.functional as F

class TaskHandler:
    def __init__(self, criterion, batches_results):
        self.criterion = criterion
        self.batches_results = batches_results
        self.best_results = dict()

    def zero_batch_results(self):
        for k,v in self.batches_results.items():
            self.batches_results[k] = np.array([])

    def compile_samples(self, samples):
        self.batches_results['samples'] = np.concatenate(
                [self.batches_results['samples'], samples])

    def compile_labels(self, labels):
        self.batches_results['ys'] = np.concatenate(
                [self.batches_results['ys'], labels])

    def format_ybatch(self, y_batch):
        return y_batch

    def compute_loss(self, model_out, y_batch):
        return self.criterion(model_out, y_batch)

    def get_sum_loss(self, loss, y_batch):
         return loss.item()*len(y_batch)


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
        batches_results = {'losses':np.array([]),
                         'correlations':np.array([]), # between outputs/labels
                         'preds':np.array([]), # the prediction for each sample
                         'ys':np.array([]), # labels of the batch
                         'samples':np.array([]) # samples of the batch
                         }

        super(RegressionHandler, self).__init__(criterion, batches_results)

        # Task name
        self.name = 'regression'


    # Ovewrite from parent class
    def format_ybatch(self, y_batch):
        return y_batch.unsqueeze(1)


    def get_label(self, f, index):
        y = np.array(f['regression_labels'][index], dtype=np.float32)

        return y


    def get_batch_results(self, model_out, y_batch):
        with torch.no_grad():
            # Pred : the output of the network for each sample
            # We .squeeze(dim=1) the model output : because model
            # output shape is batch_sizex1 and we want batch_size only
            self.batches_results['preds'] = np.concatenate(
                    [self.batches_results['preds'],
                     model_out.squeeze(dim=1).detach().cpu().numpy()])

            # Correlation
            # Pearson's r : SUM[(xi - xmean)(yi-ymean)] /
            #               SQRT[SUM[(xi-xmean)^2]*SUM[(yi-ymean)^2]]
            with torch.no_grad():
                vx =  model_out - torch.mean(model_out)
                vy = y_batch - torch.mean(y_batch)

                r = torch.sum(vx*vy) / \
                    (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

                # Multiply correlation by batch len (to account for unequal batches
                sum_r = r*y_batch.size()[0]

            self.batches_results['correlations'] = np.append(
                    self.batches_results['correlations'], sum_r.item())


    def print_baseline_results(self, baseline_results):
        # Nb of samples
        nb_samples = len(baseline_results['samples'])

        # Baseline loss
        loss = baseline_results['losses'].sum()/nb_samples

        # Baseline correlation between labels and network outputs
        corr = baseline_results['correlations'].sum()/nb_samples

        print('Baseline loss: {} baseline labels-output correlations: {}' \
              .format(loss, corr), flush=True)


    def print_epoch_results(self, train_results, valid_results):
        # Nb of samples
        nb_samples_train = len(train_results['samples'])
        nb_samples_valid = len(valid_results['samples'])

        # Loss
        loss_train = train_results['losses'].sum()/nb_samples_train
        loss_valid = valid_results['losses'].sum()/nb_samples_valid

        # Correlation between model output and labels
        corr_train = train_results['correlations'].sum()/nb_samples_train
        corr_valid = valid_results['correlations'].sum()/nb_samples_valid

        print('train loss: {} train correlation: {}'
              '\nvalid loss: {} valid correlation: {}'.format(
              loss_train, corr_train, loss_valid, corr_valid), flush=True)


    def print_resumed_best_results(self, results):
        print('best valid loss: {}'.format(results['loss']))


    def print_test_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])

        loss = results['losses'].sum()/nb_samples
        corr = results['correlations'].sum()/nb_samples

        print('Test loss: {} test correlation: {}'.format(loss, corr))


    def init_best_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])
        # Best loss
        self.best_results['loss'] = results['losses'].sum()/nb_samples


    def resume_best_results(self, results):
        self.best_results['loss'] = results['loss']


    def update_best_results(self, results):
        has_improved = False

        # Nb samples
        nb_samples = len(results['samples'])
        # Actual loss
        loss = results['losses'].sum()/nb_samples

        # Improvement if actual loss is less than best loss
        if loss < self.best_results['loss']:
            has_improved = True
            # Update best loss
            self.best_results['loss'] = loss

        return has_improved


    def save_predictions(self, results, file_fullpath):
        np.savez(file_fullpath,
                 samples=results['samples'],
                 labels=results['ys'],
                 preds=results['preds'])


