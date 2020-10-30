"""
This script creates the AttributionManager object, which handles attribution creation and analysis
"""
from pathlib import Path
import os

import numpy as np
import torch
import h5py

try:
    import seaborn as sns
    default_sns_scatter = sns.scatterplot
except ImportError:
    print('Cannot load seaborn. Continuing without visualizations...')
    default_sns_scatter = None
    pass

import pandas as pd
import matplotlib.pyplot as plt

from helpers import mainloop_utils as mlu
from Interpretability.attribution_manager import AttributionManager
from Interpretability import utils


class GraphAttributionManager(AttributionManager):
    """
    Manager of attribution plotting and visualization
    This is built from AttributionManager, so can use this to create and aggregate attributions as well
    """
    def __init__(self):
        """
        Note the correspondances between our workflow and Leo's:
        * Leo: PATH_INTGRADS, Us: self.agg_attributions
        * Leo: PATH_DATASET, Us: self.genotypes_data
        * Leo: PATH_TO_INTGRADS: leads to intgrads, feature and label names. We pass each of these seperately!
        
        Leo includes PATH_INTGRADS_2, PATH_SNPLOAD, PATH_FST, EXPERIMENT_NAME and ANALYSIS_TYPE.
        We pass these seperately in function calls!
        """
        super().__init__()
        
        self.agg_attributions = None           #  avg attributions (output of self.get_attribution_average) (numpy.array)
        self.working_dir = None                #  from this, we make paths to graph and pickles (str)
        self.feat_names = None                 #  numpy array of size (# features, ) and dtype='<U12'
        self.label_names = None                #  numpy array of size (# populations, ) and dtype='<U3'
        self.labels = None                     #  torch.tensor of size (# datapoints) and dtype=torch.int64
        
        self.GENOTYPES = [0, 1, 2]

    @property
    def graph_mode(self):
        """
        Returns true if you have everything you need to compute graphs
        (does not type check, so you can still get errors!)
        """
        return (self.working_dir is not None) and \
        (self.genotypes_data is not None) and \
        (self.agg_attributions is not None) and \
        (self.feat_names is not None) and \
        (self.label_names is not None) and \
        (self.labels is not None)
    
    
    @property
    def graph_mode_individual(self):
        """
        Returns true if you have everything you need to compute graphs (on per individual basis)
        (does not type check, so you can still get errors!)
        """
        return (self.working_dir is not None) and \
        (self.genotypes_data is not None) and \
        (self.raw_attributions_file is not None) and \
        (self.feat_names is not None) and \
        (self.label_names is not None) and \
        (self.labels is not None)
    
    def set_working_dir(self, working_dir):
        self.working_dir = working_dir
    
    def set_agg_attributions(self, agg_attributions):
        self.agg_attributions = agg_attributions
    
    def set_feat_names(self, feat_names):
        self.feat_names = feat_names.astype('str') # convert from dtype='|S12'
    
    def set_label_names(self, label_names):
        self.label_names = label_names.astype('str')
    
    def set_labels(self, labels):
        self.labels = labels
        self.label_idx = self.labels.unique()
        self.labels.to(self.device)
    
    #######################################################
    #########  Helper functions for attr plots    #########
    #########                                     #########
    #######################################################
    
    def get_genotype_freq(self, genotypes_data):
        """
        for genotypes torch.tensor([[0,1,2,0],
                                    [0,0,2,0]])
        will return (for 0) [1,0.5,0,1]
        """
        return torch.stack([(genotypes_data==i).float().mean(0) for i in self.GENOTYPES])
    
    def get_allele_freq(self, genotypes_data, ref=1):
        """
        for genotypes torch.tensor([[0,1,2,0],
                                    [0,0,2,0]])
        will return (for ref=0) [4/4, 3/4, 0/4, 4/4]
        
        We compute the frequencies w.r.t ref (1 by default)
        Note: The values are replicated to keep the same shape as get_genotype_freq
        Also Note: get_allele_freq(*, ref=1) = 1 - get_allele_freq(*, ref=0)
        """
        return (((genotypes_data == 2*ref)*2+(genotypes_data == 1)).float().mean(0)/2).repeat(len(self.GENOTYPES), 1)

    def get_metric_strat_pop(self, metric, genotypes_data, labels, global_metric=False):
        """
        Returns torch.tensor of size (# SNPs, # variants, # populations),
        If global_metric=False, each entry is metric computed for each label (population)
        If global_metric=True, the metric is computed for the global population
        Note that this is the same shape as the attrs!
        
        By default, genotypes_data is from self.genotypes_data and labels are from self.labels 
        (but you can override this, for example, if you wanted the metric to be over the entire dataset vs the test set)
        """
        
        if not global_metric:
            #  compute metric per population
            pops = [genotypes_data[labels == i] for i in labels.unique()]
            pops = [metric(pop) for pop in pops]
            pops = torch.stack(pops).permute(2, 1, 0)
        else:
            #  compute metric globally 
            pops = metric(genotypes_data).permute(1,0).unsqueeze(-1).repeat(1, 1, len(labels.unique()))
        return pops
    
    def convert_numpy_array_to_df(self, array, level_name, names=['SNP', 'Variant', 'Population']):
        """
        This converts a numpy array into a
        pandas.DataFrame with additional colums specified by names.
        
        By default, array is of size (# SNPs, # variants, # populations) 
        and the resulting pandas dataframe has columns ['SNP', 'Variant', 'Population']
        
        This is used for plotting
        """
        index = pd.MultiIndex.from_product([range(s)for s in array.shape], names=names)
        df = pd.DataFrame({level_name: array.flatten()}, index=index)

        #  converts multi-index into regular df. Easier to work with!
        return df.reset_index()

    def plot_two_arrays_against_eachother(self, arr_1, arr_2, 
                                          name_1, name_2, 
                                          row, col, 
                                          hue, save_path,  
                                          scatter_options={}, 
                                          plot_options={},
                                          sns_method=default_sns_scatter):
        """
        Given numpy arrays arr_1 and arr_2 of sizes (# SNPs, # variants, # populations),
        Each represents variabel values with names name_1 and name_2
        converts them into pandas df's with columns representing SNPs, variants and populations
        
        Proceeds to create a FacetGrid via plot_two_dfs_against_eachother
        """
        attr_1 = self.convert_numpy_array_to_df(arr_1, name_1)
        attr_2 = self.convert_numpy_array_to_df(arr_2, name_2)

        #  make plots
        self.plot_two_dfs_against_eachother(attr_1, attr_2, name_1, name_2, row, col, hue, save_path, scatter_options, plot_options, sns_method)


    def plot_two_dfs_against_eachother(self, df_1, df_2, name_1, name_2, row, col, hue, save_path, 
                                       scatter_options={}, plot_options={}, sns_method=default_sns_scatter):
        """
        Given pandas dataframes df_{1,2} with columns (SNP, Variant, Population, name_{1,2}),
        Each representing variable values with names name_1 and name_2

        Creates a FacetGrid, where we create a scatterplot of df_1[name_1] vs df_2[name_2].
        Each plots (row, column) position in the FacetGrid is based on the variable row and col respectively
        The color of the points on each scatterplot is based on hue
        """

        #  include both in single array
        df_merged = df_1.merge(df_2, on=['Variant','Population','SNP'], how='inner')
        
        #  can pass kwargs directly into facetgrid
        g = sns.FacetGrid(df_merged, row=row, col=col, hue=hue, margin_titles=True, **plot_options)
        g.map(sns_method, name_1, name_2, **scatter_options);
        #g.set(**plot_options)
        
        g.savefig(save_path)


    #######################################################
    #########         (Global) attr plots         #########
    #########                                     #########
    #######################################################

    def plot_attr_vs_metric(self, 
                            metric, 
                            metric_name,
                            genotypes_data=None, 
                            labels=None, 
                            save_path=None,
                            global_metric=False,
                            scatter_options={}, 
                            plot_options={},
                            sns_method=default_sns_scatter):
        """
        Plots attributions vs metric 
        metric is a function of the provided genotypes_data and labels
        
        If global_metric=False, the metric aggregates based on population (using labels)
        Otherwise, the metric aggegates over all the populations.
        Either way, we also aggregate over genotype (using genotype_data),
        So this returns a tensor of size (# SNPS, # variants, # populations)
    
        * Current aggegations are gene frequency and allele frequency
        
        Finally, this tensor is plotted against self.agg_attributions
        """
        if save_path is None:
            save_path = os.path.join(self.working_dir, 'attr_vs_{}.png'.format(metric_name))
        
        if genotypes_data is None:
            genotypes_data = self.genotypes_data 
        if labels is None:
            labels = self.labels
        
        if self.graph_mode:

            #  get genotype frequencies or allele frequencies
            genotype_metric = self.get_metric_strat_pop(metric, genotypes_data, labels, global_metric)
            genotype_metric = genotype_metric.cpu().numpy()

            #  make plot
            self.plot_two_arrays_against_eachother(self.agg_attributions, 
                                                   genotype_metric, 
                                                   "attributions", 
                                                   metric_name, 
                                                   row="Population", 
                                                   col="Variant", 
                                                   hue="Variant",
                                                   save_path=save_path,
                                                   scatter_options=scatter_options, 
                                                   plot_options=plot_options,
                                                   sns_method=sns_method)

    def plot_attr_vs_gene_freq(self, 
                               genotypes_data=None, 
                               labels=None, 
                               save_path=None, 
                               global_metric=False, 
                               scatter_options={}, 
                               plot_options={}, 
                               sns_method=default_sns_scatter):
        """
        Plots genotype frequency vs attributions for each population and each variant
        """

        self.plot_attr_vs_metric(self.get_genotype_freq, 
                                 "gene_frequency",
                                 genotypes_data, 
                                 labels, 
                                 save_path,
                                 global_metric,
                                 scatter_options, 
                                 plot_options,
                                 sns_method=sns_method)

    def plot_attr_vs_allele_freq(self, 
                                 genotypes_data=None, 
                                 labels=None, 
                                 ref=1, 
                                 save_path=None, 
                                 global_metric=False, 
                                 scatter_options={}, 
                                 plot_options={}, 
                                 sns_method=default_sns_scatter):
        """
        Plots allele frequency vs attributions for each population and each variant
        """

        get_allele_freq_fixed_ref = lambda x: self.get_allele_freq(x, ref=ref)

        self.plot_attr_vs_metric(get_allele_freq_fixed_ref, 
                                 "allele_frequency",
                                 genotypes_data, 
                                 labels, 
                                 save_path,
                                 global_metric,
                                 scatter_options, 
                                 plot_options,
                                 sns_method=sns_method)

    def plot_attr_vs_snp_score(self, 
                               snp_scores, 
                               score_name, 
                               save_path=None, 
                               scatter_options={}, 
                               plot_options={}, 
                               sns_method=default_sns_scatter):
        """
        Plots snp scores vs attributions for each population and each variant
        snp scores can be fst scores or  PCA loadings, etc...
        we expect snp scores to be a pd.dataframe with columns [Variant	Population	SNP, <score_name>],
        where <score_name> will be plotted against the attributions
        
        For display purposes, specify the score_name to appear on the plots (which should be <score_name>)
        """

        attrs = self.convert_numpy_array_to_df(self.agg_attributions, 'attributions')

        self.plot_two_dfs_against_eachother(attrs, 
                                            snp_scores, 
                                            'attributions', 
                                            score_name, 
                                            "Population", 
                                            "Variant", 
                                            "Variant", 
                                            save_path, 
                                            scatter_options, 
                                            plot_options,
                                            sns_method=sns_method)


    def plot_attr_histograms(self, 
                             save_path=None, 
                             plot_options={'ylim': (0, 50000)}, 
                             scatter_options={'bins': np.arange(-1e-4, 1e-4, 1e-6)},
                             pops_to_view=[0, 1]):
        """
        Plots histogram of snp attributions (per variant per population)
        """
        if self.graph_mode:
            if save_path is None:
                save_path = os.path.join(self.working_dir, 'attr_hist.png')

            attrs = self.convert_numpy_array_to_df(self.agg_attributions, 'attributions')

            #  can pass kwargs directly into facetgrid
            g = sns.FacetGrid(attrs[(attrs.Population.isin(pops_to_view))], row='Population', col='Variant', margin_titles=True, **plot_options)
            g.map(sns.distplot, 'attributions', **scatter_options);
            g.set(**plot_options)
            g.set_xticklabels(rotation=45)

            g.savefig(save_path)

    #######################################################
    #########       (Individual) attr plots       #########
    #########                                     #########
    #######################################################

    def plot_attr_individuals(self,
                              save_path=None, 
                              plot_options={'ylim': (0, 2000)}, 
                              scatter_options={'bins': np.arange(-1e-4, 1e-4, 1e-6), 'kde': False},
                              pops_to_view=[0, 1, 2, 3, 4],
                              indv_to_view=None):

        if self.graph_mode_individual:
            if save_path is None:
                save_path = os.path.join(self.working_dir, 'attr_hist_indv.png')

            with h5py.File(self.raw_attributions_file, 'r') as hf:
                self.attr_type = list(hf.keys())[0]
                n_categories = hf[self.attr_type].shape[2]
                if indv_to_view is None:
                    size = 1
                    indv_to_view = np.random.choice(hf[self.attr_type].shape[0], size=size)
                else:
                    size = len(indv_to_view)
                data_to_plot = np.concatenate([hf[self.attr_type][i].reshape(1, -1, n_categories) for i in indv_to_view])
                attrs = self.convert_numpy_array_to_df(data_to_plot, 'attributions', names=['individual', 'SNP', 'Population'])
                attrs['Variant'] = np.repeat(self.genotypes_data[indv_to_view].flatten().cpu().numpy(), n_categories)

            #  can pass kwargs directly into facetgrid
            g = sns.FacetGrid(attrs[(attrs.Population.isin(pops_to_view))], row='Population', col='Variant', margin_titles=True, **plot_options)
            g.map(sns.distplot, 'attributions', **scatter_options);
            g.set(**plot_options)
            g.set_xticklabels(rotation=45)

            g.savefig(save_path)
