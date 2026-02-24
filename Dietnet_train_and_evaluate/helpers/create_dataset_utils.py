import numpy as np

# Load genotypes of a sample from a line in the genotype file
def parse_genotypes(line):
    # Line : Sample id and genotype values across all SNPs
    sample = (line.split('\t')[0]).strip()

    # Fill with genotypes of all SNPs for the individual
    genotypes = []
    for i in line.split('\t')[1:]:
        # Replace missing values with -1
        if i.strip() == './.' or i.strip() == 'NA':
            genotype = -1
        else:
            genotype = int(i.strip())

        genotypes.append(genotype)

    genotypes = np.array(genotypes, dtype='int8')

    return sample, genotypes


# Load labels of all samples from the label file
def load_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    mat = np.array([l.strip('\n').split('\t') for l in lines])

    samples = mat[1:,0]
    labels = mat[1:,1]

    return samples, labels


# Order labels to match the provided list of samples
def order_labels(samples, samples_in_labels, labels):
    idx = [np.where(samples_in_labels == s)[0][0] for s in samples]

    return np.array([labels[i] for i in idx])


# Convert class labels to numerical values
def numeric_encode_labels(labels):
    label_names = np.sort(np.unique(labels))

    encoded_labels = [np.where(label_names==i)[0][0] for i in labels]

    return label_names, encoded_labels
