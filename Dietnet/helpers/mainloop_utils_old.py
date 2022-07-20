import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.profiler as profiler

from helpers import model
from helpers import dataset_utils as du


def train_step(mod_handler, device, optimizer, train_generator,
               set_size, criterion, mus, sigmas, task, normalize):
    # Monitoring set up : Minibatch
    minibatch_loss = []
    minibatch_n_right = [] # nb of good classifications

    # Looping through minibatches
    for x_batch, y_batch, _ in train_generator:
        with profiler.record_function('To device'):
            # Send data to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.float()

            if task == 'regression':
                y_batch = y_batch.unsqueeze(1)

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)

        # Reset optimizer
        optimizer.zero_grad()


        # Forward pass
        model_out = mod_handler.forwardpass(x_batch)

        # Compute loss (softmax computation done in loss)
        loss = criterion(model_out, y_batch)

        # Compute gradients
        loss.backward()

        # Optimize
        optimizer.step()

        # Monitoring : Minibatch
        minibatch_loss.append(loss.item()) # mean loss of the minibatch

        # Classification: keep nb of good predictions for accuracy computation
        if task == 'classification':
            _, pred = get_predictions(model_out) # softmax computation
            minibatch_n_right.append(((y_batch - pred) ==0).sum().item())

        #batch_time = time.time() - batch_start_time
        #print('Batch time:', batch_time, flush=True)

        """
        if comb_model.training:
            with torch.no_grad():
                filename = 'COMPARE/train_loop_regress.pt'
                torch.save(d, filename)
        break
        """

    # Monitoring: Epoch
    epoch_loss = np.array(minibatch_loss).mean()

    if task == 'classification':
        epoch_acc = np.array(minibatch_n_right).sum() / float(set_size)*100
        epoch_result = (epoch_loss, epoch_acc)

    elif task == 'regression':
        epoch_result = (epoch_loss,)

    return epoch_result



def eval_step(mod_handler, device, valid_generator,
              set_size, criterion, mus, sigmas, task, normalize):

    # Monitoring: Minibatch setup
    minibatch_loss = []
    minibatch_n_right = [] # nb of good classifications

    test_pred = torch.tensor([]).to(device) #prediction of each sample
    test_ys = np.array([])
    test_samples = np.array([]) #test samples

    for batch, (x_batch, y_batch, samples) in enumerate(valid_generator):
        # Save samples
        test_samples = np.concatenate([test_samples, samples])
        # Save labels
        test_ys = np.concatenate([test_ys, y_batch])

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        if task == 'regression':
            y_batch = y_batch.unsqueeze(1)

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        model_out = mod_handler.forwardpass(x_batch)

        # Loss
        loss = criterion(model_out, y_batch)

        # Monitoring : Minibatch
        weighted_loss = loss.item()*len(y_batch) # for unequal minibatches
        minibatch_loss.append(weighted_loss)

        # Classification: keep nb of good predictions for accuracy computation
        if task == 'classification':
            _, pred = get_predictions(model_out) # softmax computation
            minibatch_n_right.append(((y_batch - pred) ==0).sum().item())

        elif task == 'regression':
            test_pred = torch.cat((test_pred,comb_model_out.detach()), dim=0)


    epoch_loss = np.array(minibatch_loss).sum()/set_size
    #epoch_loss = np.array(minibatch_loss).mean()

    if task == 'classification':
        epoch_acc = np.array(minibatch_n_right).sum() / float(set_size)*100
        epoch_result = (epoch_loss, epoch_acc)

    elif task == 'regression':
        epoch_result = (epoch_loss,test_pred,test_ys,test_samples)

    return epoch_result


def test_step(mod_handler, device, test_generator,
        set_size, criterion, mus, sigmas, task, normalize):
    # Saving data seen while looping through minibatches
    minibatch_loss = []
    minibatch_n_right = [] #number of good classifications
    test_pred = torch.tensor([]).to(device) #prediction of each sample
    test_score = torch.tensor([]).to(device) #softmax values of each sample
    test_samples = np.array([]) #test samples
    test_ys = np.array([]) #true labels of test samples

    for i, (x_batch, y_batch, samples) in enumerate(test_generator):
        # Save samples
        test_samples = np.concatenate([test_samples, samples])
        # Save labels
        test_ys = np.concatenate([test_ys, y_batch])

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        if task == 'regression':
            y_batch = y_batch.unsqueeze(1)

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        model_out = mod_handler.forwardpass(x_batch)

        # Loss
        loss = criterion(model_out, y_batch)

        # Monitoring : Minibatch
        weighted_loss = loss.item()*len(y_batch) # for unequal minibatches
        minibatch_loss.append(weighted_loss)

        # Predictions
        if task == 'classification':
            score, pred = get_predictions(model_out)
            test_pred = torch.cat((test_pred,pred), dim=-1)
            test_score = torch.cat((test_score,score), dim=0)

            # Nb of good classifications for the minibatch
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

        elif task == 'regression':
            test_pred = torch.cat((test_pred, model_out.detach()), dim=0)

    test_loss = np.array(minibatch_loss).sum()/set_size
    #test_loss = np.array(minibatch_loss).mean()

    # Test results to return
    if task == 'classification':
        # Total accuracy
        test_acc = np.array(minibatch_n_right).sum() / float(set_size)*100

        test_results = (test_score, test_pred, test_acc)

    elif task == 'regression':
        # Pearson correlation coefficient
        print('Computing Pearson correlation coefficient', flush=True)
        r = compute_correlation(
                test_pred,
                torch.from_numpy(test_ys).to(device).unsqueeze(1)
                )
        test_results = (test_loss, test_pred, r)

    return test_samples, test_ys, test_results


def get_last_layers(comb_model, device, test_generator, set_size,
                    mus, sigmas, emb, task):
    # Saving data seen while looping through minibatches
    minibatch_n_right = [] #number of good classifications
    test_pred = torch.tensor([]).to(device) #prediction of each sample
    test_score = torch.tensor([]).to(device) #softmax values of each sample
    test_samples = np.array([]) #test samples
    test_ys = np.array([]) #true labels of test samples

    before_last_layer = torch.tensor([])
    out_layer = torch.tensor([])

    for i, (x_batch, y_batch, samples) in enumerate(test_generator):
        # Save samples
        test_samples = np.concatenate([test_samples, samples])
        # Save labels
        test_ys = np.concatenate([test_ys, y_batch])

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        comb_model_before_last, comb_model_out = comb_model(emb, x_batch, save_layers=True)

        # Save layers
        before_last_layer = torch.cat((before_last_layer,comb_model_before_last.detach().cpu()),dim=0)
        out_layer = torch.cat((out_layer,comb_model_out.detach().cpu()), dim=0)

        # Predictions
        if task == 'classification':
            score, pred = get_predictions(comb_model_out)
            test_pred = torch.cat((test_pred,pred), dim=-1)
            test_score = torch.cat((test_score,score), dim=0)

            # Nb of good classifications for the minibatch
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

        elif task == 'regression':
            test_pred = torch.cat((test_pred,comb_model_out.detach()), dim=-1)

    # Total accuracy
    test_acc = 0.0
    if task == 'classification':
        test_acc = np.array(minibatch_n_right).sum() / float(set_size)*100

    return test_samples, test_ys, test_score, test_pred, test_acc, before_last_layer, out_layer



def get_predictions(model_output):
    with torch.no_grad():
        score = F.softmax(model_output, dim=1)
        _, pred = torch.max(score, dim=1)

    return score, pred


def compute_correlation_np(x, y):
    # Pearson's r : SUM[(xi - xmean)(yi-ymean)] / SQRT[SUM[(xi-xmean)^2]*SUM[(yi-ymean)^2]]
    vx = x - np.mean(x)
    vy = y - np.mean(y)

    r = np.sum(vx*vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    return r


def compute_correlation(x, y):
    # Pearson's r : SUM[(xi - xmean)(yi-ymean)] / SQRT[SUM[(xi-xmean)^2]*SUM[(yi-ymean)^2]]
    with torch.no_grad():
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        r = torch.sum(vx*vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

    return r.item()


def has_improved(best_result, actual_result):
    # Classification
    if len(best_result) == 2:
        # Improvement if actual acc is greater than best acheived acc
        if actual_result[1] > best_result[1]:
            return True
        # Improvement if acc is same as best acc and loss is min loss achieve
        if actual_result[1] == best_result[1] and actual_result[0] < best_result[0]:
            return True

        # No improvement
        return False

    # Regression
    elif len(best_result) == 1:
        # Improvement if loss is min loss achieve
        if actual_result[0] < best_result[0]:
            return True

        # No improvement
        return False


def update_best_result(best_result, actual_result):
    # Classification
    if len(best_result) == 2:
        # Accuracy
        if actual_result[1] > best_result[1]:
            updated_acc = actual_result[1]
        else:
            updated_acc = best_result[1]

        # Loss
        if actual_result[0] < best_result[0]:
            updated_loss = actual_result[0]
        else:
            updated_loss = best_result[0]

        return (updated_loss, updated_acc)

    # Regression
    if len(best_result) == 1:
        # Loss
        if actual_result[0] < best_result[0]:
            updated_loss = actual_result[0]
        else:
            updated_loss = best_result[0]
            print('Warning with updated loss')

        return (updated_loss,)


def has_improved_old(best_acc, actual_acc, min_loss, actual_loss):
    if actual_acc > best_acc:
        return True
    if actual_acc == best_acc and actual_loss < min_loss:
        return True

    return False


def train_step_mlp(mlp, device, optimizer, train_generator, set_size,
                   criterion, mus, sigmas, task):
    # Monitoring : Minibatch setup
    minibatch_loss = []
    minibatch_n_right = [] # nb of good classifications

    for x_batch, y_batch, _ in train_generator:
        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        x_batch = du.normalize(x_batch, mus, sigmas)

        # Reset optimizer
        optimizer.zero_grad()

        # Forward pass
        mlp_out = mlp(x_batch)

        # Get prediction
        if task == 'classification':
            # Softmax computation
            _, pred = get_predictions(mlp_out)

        # Compute loss (softmax computation done in loss if classification)
        loss = criterion(mlp_out, y_batch)

        # Compute gradients
        loss.backward()

        # Optimize
        optimizer.step()

        # Monitoring : Minibatch
        minibatch_loss.append(loss.item()) # mean loss of the minibatch
        if task == 'classification':
            minibatch_n_right.append(((y_batch - pred) ==0).sum().item())

    # Monitoring: Epoch
    epoch_loss = np.array(minibatch_loss).mean()
    epoch_acc = 0.0
    if task == 'classification':
        epoch_acc = np.array(minibatch_n_right).sum() / float(set_size)*100

    return epoch_loss, epoch_acc


def eval_step_mlp(mlp, device, valid_generator, set_size,
                  criterion, mus, sigmas, task):
    # Monitoring: Minibatch setup
    minibatch_loss = []
    minibatch_n_right = [] # nb of good classifications

    for x_batch, y_batch, _ in valid_generator:
        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        mlp_out = mlp(x_batch)

        # Predictions
        if task == 'classification':
            _, pred = get_predictions(mlp_out)

        # Loss
        loss = criterion(mlp_out, y_batch)

        # Monitoring : Minibatch
        weighted_loss = loss.item()*len(y_batch) # for unequal minibatches
        minibatch_loss.append(weighted_loss)
        if task == 'classification':
            minibatch_n_right.append(((y_batch - pred) ==0).sum().item())

    epoch_loss = np.array(minibatch_loss).sum()/set_size
    epoch_acc = 0.0
    if task == 'classification':
        epoch_acc = np.array(minibatch_n_right).sum() / float(set_size)*100

    return epoch_loss, epoch_acc

def test_step_mlp(mlp, device, test_generator, set_size,
              mus, sigmas, task):
    # Saving data seen while looping through minibatches
    minibatch_n_right = [] #number of good classifications
    test_pred = torch.tensor([]).to(device) #prediction of each sample
    test_score = torch.tensor([]).to(device) #softmax values of each sample
    test_samples = np.array([]) #test samples
    test_ys = np.array([]) #true labels of test samples

    for i, (x_batch, y_batch, samples) in enumerate(test_generator):
        # Save samples
        test_samples = np.concatenate([test_samples, samples])
        # Save labels
        test_ys = np.concatenate([test_ys, y_batch])

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        mlp_out = mlp(x_batch)

        # Predictions
        if task == 'classification':
            score, pred = get_predictions(mlp_out)
            test_pred = torch.cat((test_pred,pred), dim=-1)
            test_score = torch.cat((test_score,score), dim=0)

            # Nb of good classifications for the minibatch
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

        elif task == 'regression':
            test_pred = torch.cat((test_pred,mlp_out), dim=-1)

    # Total accuracy
    test_acc = 0.0
    if task == 'classification':
        test_acc = np.array(minibatch_n_right).sum() / float(set_size)*100

    return test_samples, test_ys, test_score, test_pred, test_acc


def create_disc_model(comb_model, emb, device):
    """
    this only works with 1 GPU or CPU
    note: the function name is misleading. This does not create a new model. It just
    takes the pre-existing model and
    returns a function that performs the forward pass on it (with fixed embedding).
    this should be okay since python passes args by reference, so even if comb_model
    weights change, the corresponding function will also change

    Finally, remember that disc_model will be in whatever mode comb_model is,
    so if comb_model is in train mode, switch it to eval mode before calling
    ßthe output of this.
    """

    if torch.cuda.device_count() > 1:
        print('warning: this only works during training/inference with 1 GPU!')
    comb_model = comb_model.eval()
    comb_model.to(device)
    discrim_model = lambda x: comb_model(emb, x) # recreate discrim_model
    return discrim_model


def create_disc_model_multi_gpu(comb_model, emb, device, eps=1e-5, incl_softmax=False):
    """
    Transforms comb_model + emb into equivalent discrim model (with fatlayer weights added as parameters)
    This model can now be sent to multiple GPUs without any bugs
    (cannot do multi-GPU with comb_model since dataparallel will attempt to split the embedding up,
    which will result in size incompatibilities)

    Must pass batchnorm eps seperately in case loading from Theano model!
    Must pass incl_softmax seperately in case loading from Theano model!
    """

    n_feats_emb = emb.size()[1] # input of aux net
    n_feats = emb.size()[0] # input of main net
    # Hidden layers size
    emb_n_hidden_u = 100
    discrim_n_hidden1_u = 100
    discrim_n_hidden2_u = 100

    #  put in eval mode and send to correct device
    comb_model = comb_model.eval().to(device)
    emb = emb.to(device)
    fatLayer_weights = torch.transpose(comb_model.feat_emb(emb),1,0)

    #  initialize discriminitive network with fatlayer as a parameter
    #  send back to cpu while loading weights
    disc_net = model.Discrim_net(fatLayer_weights,
                                 n_feats=n_feats_emb,
                                 n_hidden1_u=discrim_n_hidden1_u,
                                 n_hidden2_u=discrim_n_hidden2_u,
                                 n_targets=26,
                                 eps=eps,
                                 incl_softmax=incl_softmax)

    #  copy over all weights
    disc_net.out.weight.data = comb_model.disc_net.out.weight.data
    disc_net.out.bias.data = comb_model.disc_net.out.bias.data
    disc_net.bn2.weight.data = comb_model.disc_net.bn2.weight.data
    disc_net.bn2.bias.data = comb_model.disc_net.bn2.bias.data
    disc_net.bn2.running_mean = comb_model.disc_net.bn2.running_mean
    disc_net.bn2.running_var = comb_model.disc_net.bn2.running_var
    disc_net.hidden_2.weight.data = comb_model.disc_net.hidden_2.weight.data
    disc_net.hidden_2.bias.data = comb_model.disc_net.hidden_2.bias.data
    disc_net.bn1.weight.data = comb_model.disc_net.bn1.weight.data
    disc_net.bn1.bias.data = comb_model.disc_net.bn1.bias.data
    disc_net.bn1.running_mean = comb_model.disc_net.bn1.running_mean
    disc_net.bn1.running_var = comb_model.disc_net.bn1.running_var
    disc_net.hidden_1.bias.data = comb_model.disc_net.fat_bias.data

    assert (disc_net.hidden_1.weight.data == fatLayer_weights).all()
    disc_net = disc_net.eval().to('cpu')

    #  Now we can do dataparallel
    if torch.cuda.device_count() > 1:
        disc_net = nn.DataParallel(disc_net)
        print("{} GPUs detected! Running in DataParallel mode".format(torch.cuda.device_count()))
    disc_net.to(device)
    return disc_net

def convert_theano_array_to_pytorch_tensor(tensor, array):
    array_as_tensor = torch.from_numpy(array.T)
    assert tensor.data.shape == array_as_tensor.shape
    tensor.data = array_as_tensor

def convert_theano_array_to_pytorch_tensor_1d(tensor, array):
    array_as_tensor = torch.from_numpy(array)
    assert tensor.data.shape == array_as_tensor.shape
    tensor.data = array_as_tensor

def convert_theano_bn_pytorch(bn, theano_arr_1, theano_arr_2, theano_arr_3, theano_arr_4):
    # Idea from: https://discuss.pytorch.org/t/is-there-any-difference-between-theano-convolution-and-pytorch-convolution/10580/10
    # Specifically how to load BN from theano to pytorch
    #extractor += [nn.BatchNorm2d(in_channel)]
    #extractor[-1].weight.data = torch.from_numpy(parameters[i-3])
    #extractor[-1].bias.data = torch.from_numpy(parameters[i-2])
    #extractor[-1].running_mean = torch.from_numpy(parameters[i-1])
    #extractor[-1].running_var = torch.from_numpy((1./(parameters[i]**2)) - 1e-4)

    convert_theano_array_to_pytorch_tensor_1d(bn.weight, theano_arr_2)
    convert_theano_array_to_pytorch_tensor_1d(bn.bias, theano_arr_1)
    convert_theano_array_to_pytorch_tensor_1d(bn.running_mean, theano_arr_3)
    convert_theano_array_to_pytorch_tensor_1d(bn.running_var, (1./(theano_arr_4**2)) - 1e-4)


def load_theano_model(n_feats_emb, emb_n_hidden_u, discrim_n_hidden1_u, discrim_n_hidden2_u, n_targets, theano_weight_file, device, only_discrim_model=True):
    #  loads theano weights file into PyTorch's comb_net OR Discrim_net

    theano_model_params = np.load(theano_weight_file)
    print('theano layer shapes:')
    for f in theano_model_params.files:
        print(f, theano_model_params[f].shape)

    comb_model = model.CombinedModel(
                 n_feats=n_feats_emb,
                 n_hidden_u=emb_n_hidden_u,
                 n_hidden1_u=discrim_n_hidden1_u,
                 n_hidden2_u=discrim_n_hidden2_u,
                 n_targets=n_targets,
                 param_init=None,
                 eps=1e-4, # Theano uses 1e-4 for batch norm instead of PyTorch default of 1e-5
                 input_dropout=0.,
                 incl_softmax=True) # theano includes softmax in output

    #  feat emb model
    convert_theano_array_to_pytorch_tensor(comb_model.feat_emb.hidden_1.weight, theano_model_params['arr_0'])
    convert_theano_array_to_pytorch_tensor(comb_model.feat_emb.hidden_2.weight, theano_model_params['arr_1'])

    #  embedding
    emb = torch.tensor(theano_model_params['arr_2'])

    #  disc model
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.fat_bias, theano_model_params['arr_3'])

    convert_theano_bn_pytorch(comb_model.disc_net.bn1,
                          theano_model_params['arr_4'],
                          theano_model_params['arr_5'],
                          theano_model_params['arr_6'],
                          theano_model_params['arr_7'])

    convert_theano_array_to_pytorch_tensor(comb_model.disc_net.hidden_2.weight, theano_model_params['arr_8'])
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.hidden_2.bias, theano_model_params['arr_9'])

    convert_theano_bn_pytorch(comb_model.disc_net.bn2,
                              theano_model_params['arr_10'],
                              theano_model_params['arr_11'],
                              theano_model_params['arr_12'],
                              theano_model_params['arr_13'])

    convert_theano_array_to_pytorch_tensor(comb_model.disc_net.out.weight, theano_model_params['arr_14'])
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.out.bias, theano_model_params['arr_15'])

    model_to_return = comb_model.to(device)

    if only_discrim_model:

        emb = emb.to(device)
        #  create disc_net from loaded comb_model
        model_to_return = create_disc_model_multi_gpu(model_to_return,
                                                      emb, device,
                                                      eps=1e-4, # Theano uses 1e-4 for batch norm instead of PyTorch default of 1e-5
                                                      incl_softmax=True) # Theano includes softmax in model

    return model_to_return


def load_model(model_path,
               emb,
               device,
               n_feats,
               n_hidden_u_aux,
               n_hidden_u_main,
               n_targets,
               input_dropout,
               incl_bias=True,
               incl_softmax=False,
               load_comb_model=False):
    """
    Load (discrim) model for test time / attribution computation
    Set load_comb_model to True to load the combined model instead
    """
    comb_model = model.CombinedModel(
        n_feats,
        n_hidden_u_aux,
        n_hidden_u_main,
        n_targets,
        param_init=None,
        input_dropout=input_dropout,
        incl_bias=incl_bias,
        incl_softmax=incl_softmax)

    comb_model.load_state_dict(torch.load(model_path))
    comb_model.to(device)
    comb_model = comb_model.eval()

    if load_comb_model:
        return comb_model

    else:
        #discrim_model = create_disc_model_multi_gpu(comb_model, emb, device, incl_softmax)
        discrim_model = create_disc_model(comb_model, emb, device)

        del comb_model
        torch.cuda.empty_cache()

        return discrim_model
