import numpy as np

import torch


def put_noise():
    pass

def compute_reconstructed_seq(output, x_batch):
    """
    The output of the decoder is of size seq_len*nb_classes
    In the genotypes case, we have an output of nb_snps*3
    (3 is nb of genotypes: 0,1,2

    Here we compute the reconstruction of the input seq:
    We compute the softmax over neurones associated with the same
    element of the input sequence to get the decoder prediction
    for that element. The final prediction for an element is the
    max of the values computed by the softmax

    Inputs :
        - output: netork (decoder) output
        - x_batch is the input seq
    Return :
        The reconstructed seq from decoder output for all samples
    """
    # Reshape reconstruction to get batch_size x nb_classes x seq_len
    # each column is a position with neurones for genotypes 0, 1 and 2
    this_batch_size = output.size()[0]
    output_len = output.size()[-1]
    seq_len = x_batch.size()[-1]

    nb_classes = int(output_len/seq_len)

    output_shaped = output.reshape(this_batch_size, seq_len, nb_classes).permute(0,2,1)

    # Get predicted class (genotype) per position
    with torch.no_grad():
        # compute softmax along each col
        softmax_fn = torch.nn.Softmax(dim=1)
        softmax_out = softmax_fn(output_shaped)

    # Get the predicted class per col
    reconstruction = torch.argmax(softmax_out, dim=1)

    return reconstruction


def train_step(results_fullpath, train_generator, encoder, decoder,
               optimizer_encoder, optimizer_decoder, criterion, device):
    # Stuff to compile over batches
    batches_samples = np.array([])
    batches_inputs = np.array([])
    batches_reconstructions = np.array([])
    batches_n_right = np.array([])

    batches_losses = []
    for x_batch, sample_batch in train_generator:
        # Compile samples
        batches_samples = np.concatenate([batches_samples, sample_batch])
        # Compile labels
        batches_inputs = np.vstack([batches_inputs, x_batch]) \
                if batches_inputs.size else x_batch

        x_batch = x_batch.to(device).float()

        # Reset optimizer
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        # Forward pass
        bottleneck = encoder(x_batch)
        output = decoder(bottleneck)

        # Loss
        # Reshape input for loss computation
        this_batch_size = output.size()[0]
        output_len = output.size()[-1]
        seq_len = x_batch.size()[-1]
        nb_classes = int(output_len/seq_len)

        # x_batch is converted to long tensor so that genotypes
        # become classes for the loss computation
        loss = criterion(output.reshape(this_batch_size, seq_len, nb_classes).permute(0,2,1), x_batch.long())

        # Compile batch loss
        # (multiplication by batch size to account for unequal batches)
        batches_losses.append(loss.item()*this_batch_size)

        # Optimization
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        # Compute reconstructed seq
        reconstruction = compute_reconstructed_seq(output, x_batch)
        # Reconstruction accuracy
        n_right_positions = (reconstruction == x_batch).sum().item()
        batches_n_right = np.append(batches_n_right, n_right_positions)

        # Compile reconstructions
        batches_reconstructions = np.vstack(
                [batches_reconstructions, reconstruction.cpu().numpy()]) \
                        if batches_reconstructions.size \
                        else reconstruction.cpu().numpy()

    # Epoch loss is the loss sum over batches and averaged over all samples
    epoch_loss = np.array(batches_losses).sum() / len(batches_samples)
    # Epoch reconstruction acc is nb of right positions / total nb positions
    epoch_reconstruction_acc = batches_n_right.sum() / batches_reconstructions.size

    # Save reconstruction
    np.savez(results_fullpath,
             samples=batches_samples,
             inputs=batches_inputs,
             reconstruction=batches_reconstructions)

    return epoch_loss, epoch_reconstruction_acc


def valid_step(results_fullpath, valid_generator, encoder, decoder,
               criterion, device):
    # Stuff to compile over batches
    batches_samples = np.array([])
    batches_inputs = np.array([])
    batches_reconstructions = np.array([])
    batches_n_right = np.array([])


    batch_losses = []
    for x_batch, sample_batch in valid_generator:
        # Compile samples
        batches_samples = np.concatenate([batches_samples, sample_batch])
        # Compile labels
        batches_inputs = np.vstack([batches_inputs, x_batch]) \
                if batches_inputs.size else x_batch

        x_batch = x_batch.to(device).float()

        # Forward pass
        bottleneck = encoder(x_batch)
        output = decoder(bottleneck)

        # Loss
        # Reshape input for loss computation
        this_batch_size = output.size()[0]
        output_len = output.size()[-1]
        seq_len = x_batch.size()[-1]
        nb_classes = int(output_len/seq_len)

        loss = criterion(output.reshape(this_batch_size, seq_len, nb_classes).permute(0,2,1), x_batch.long())

        # Compile batch loss
        # (multiplication by batch size to account for unequal batches)
        batch_losses.append(loss.item()*this_batch_size)

        # Compute reconstructed seq by decoder
        reconstruction = compute_reconstructed_seq(output, x_batch)
        # Reconstruction accuracy
        n_right_positions = (reconstruction == x_batch).sum().item()
        batches_n_right = np.append(batches_n_right, n_right_positions)

        # Compile reconstructions
        batches_reconstructions = np.vstack(
                [batches_reconstructions, reconstruction.cpu().numpy()]) \
                        if batches_reconstructions.size \
                        else reconstruction.cpu().numpy()

    # Epoch loss is the loss sum over batches and averaged over all samples
    epoch_loss = np.array(batch_losses).sum()/len(batches_samples)
    # Epoch reconstruction acc is nb of right positions / total nb positions
    epoch_reconstruction_acc = batches_n_right.sum() / batches_reconstructions.size

    # Save reconstruction
    np.savez(results_fullpath,
             samples=batches_samples,
             inputs=batches_inputs,
             reconstruction=batches_reconstructions)

    return epoch_loss, epoch_reconstruction_acc


