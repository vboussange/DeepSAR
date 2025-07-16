import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import seaborn as sns
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from deepsar.cld import create_comp_matrix_allpair_t_test, multcomp_letters
from matplotlib.colors import LinearSegmentedColormap

RCPARAMS_DICT = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "lines.markersize": 3
}

COLOR_PALETTE = [(0.09019607843137255, 0.41568627450980394, 0.050980392156862744, 1.0),
                        (0.08627450980392157, 0.23921568627450981, 0.07058823529411765, 1.0),
                        (0.8274509803921568, 0.7372549019607844, 0.14901960784313725, 1.0),
                        (0.7215686274509804, 0.8313725490196079, 0.10196078431372549, 1.0),
                        (0.06666666666666667, 0.4, 0.6196078431372549),
                        (0.11372549019607843, 0.8470588235294118, 0.6274509803921569, 1.0),
                        (0.3803921568627451, 0.0196078431372549, 0.0196078431372549, 1.0),
                        (0.5450980392156862, 0.03529411764705882, 0.03529411764705882, 1.0),
                        (0.5529411764705883, 0.6274509803921569, 0.796078431372549)]

COLORS_BR = ["#f72585","#b5179e","#7209b7","#560bad","#480ca8","#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"]
# check https://coolors.co/palettes/popular/gradient
CMAP_BR = LinearSegmentedColormap.from_list("species_richness", COLORS_BR[::-1])


COLORS_DSR = ["#00296b","#003f88","#00509d","#fdc500","#ffd500"]
CMAP_DSR = LinearSegmentedColormap.from_list("dsr", COLORS_DSR)

COLORS_GO = ["#ff9f1c", "#ffbf69", "#ffffff", "#cbf3f0", "#2ec4b6"]
CMAP_GO = LinearSegmentedColormap.from_list("custom_cmap", COLORS_GO)


def boxplot_bypreds(result_modelling=None,
                    ax=None,
                    spread=None,
                    colormap=None,
                    legend=False,
                    xlab=None,
                    ylab=None,
                    yscale="log",
                    yname="test_neg_mean_squared_error",
                    habitats = None,
                    predictors = None,
                    widths=0.1,
                    alpha=0.05,
                    predictor_labels = None,
                    cld=True): #significance of p value
    
    if not predictor_labels:
        predictor_labels = predictors

    if not habitats:
        habitats = list(result_modelling.keys())
    N = len(predictors)  # number of habitats
    color_palette = sns.color_palette(colormap, N)
    M = len(habitats)  # number of groups
    for j, p in enumerate(predictors):
        y = [
            result_modelling[hab][p][yname] for hab in habitats
        ]
        xx = np.arange(1, M + 1) + (
            np.linspace(0,1, N)[j] - 0.5
        ) * spread  # artificially shift the x values to better visualise the std
        boxplot(ax, y, xx, color_palette[j], widths)

    if cld:
        for i, hab in enumerate(habitats):
            print(hab)       
            # calculating paired t test
            data = [
                result_modelling[hab][p][yname] for p in predictors
            ]
            groups = [np.repeat(predictors[i],len(d)) for (i,d) in enumerate(data)]
            mc = MultiComparison(np.concatenate(data), np.concatenate(groups))
            test_results = mc.allpairtest(stats.ttest_rel, alpha=alpha)
            
            print(test_results[0])
            comp_matrix = create_comp_matrix_allpair_t_test(test_results)
            print(comp_matrix)
            
            letters = multcomp_letters(comp_matrix < alpha)
            print(letters)

            xx = i + 1 + (
                np.linspace(0, 1, N) - 0.5
            ) * spread
            for (j,p) in enumerate(predictors):
                if p in letters:
                    ypos = min(calculate_whisker_pos(data[j]), ax.get_ylim()[1])
                    ax.text(xx[j], ypos, letters[p], ha='center', va='bottom', fontsize=8, color='black')

        
    ax.set_ylabel(ylab)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlab)
    x = habitats
    ax.set_xticks(np.arange(1, len(x) + 1))
    ax.set_xticklabels(x)
    if legend:
        ax.legend(handles=[
            Line2D([0], [0], color=color_palette[i], label=predictor_labels[i])
            for i in range(len(predictors))
        ])
        plt.show()

def boxplot(ax, y, positions, color, widths):
    bplot = ax.boxplot(
        y,
        positions=positions,
        showfliers=False,
        widths=widths,
        vert=True,  # vertical box alignment
        patch_artist=True)  # fill with color

    for patch in bplot['boxes']:
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    for item in ['caps', 'whiskers']:
        for element in bplot[item]:
            element.set_color(color)
    for element in bplot["medians"]:
        element.set_color("black")
        
def calculate_whisker_pos(data, whis=1.5):
    # Calculate Q1 and Q3
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate Q3 + 1.5 * IQR
    upper_bound = Q3 +whis * IQR
    
    # Find the highest datum below the upper bound
    upper_whisker_candidates = data[data <= upper_bound]

    # Handle case where no data points are below the upper bound
    if len(upper_whisker_candidates) == 0:
        upper_whisker = Q3
    else:
        upper_whisker = np.max(upper_whisker_candidates)

    return upper_whisker
