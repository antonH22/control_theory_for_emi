import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

F = 1

params = {}
# Figure Layout
params['figure.constrained_layout.use'] = True
params['figure.constrained_layout.h_pad'] = 0   #inches
params['figure.constrained_layout.w_pad'] = 0   #inches
# Spines and Edges
params['axes.spines.bottom'] = True
params['axes.spines.left'] = True
params['axes.spines.right'] = True
params['axes.spines.top'] = True
params['axes.linewidth'] = 0.25*F       #points
params['axes.edgecolor'] = 'black'
# Grid
params['axes.grid'] = False
params['axes.grid.axis'] = 'y'
params['grid.color'] = 'grey'
params['grid.linewidth'] = 0.15*F       #points
# Lines
params['lines.linewidth'] = 1*F         #points
params['lines.markersize'] = 3*F
params['errorbar.capsize'] = 2*F        #pixels    
params['axes.prop_cycle'] = mpl.cycler('color', ['slategrey'])
# General Text
params['font.size'] = (10*F)            #points
params['font.family'] = 'sans-serif'
params['font.serif'] = 'cmr10'
params['font.sans-serif'] = 'arial'
params['text.color'] = 'black'
params['text.parse_math'] = True
params['mathtext.fontset'] = 'cm'
# Title
params['figure.titlesize'] = 'medium'
params['axes.titlesize'] = 'medium'
# Axis Labels
params['axes.labelcolor'] = 'black'
params['axes.labelsize'] = 'small'
params['axes.formatter.use_mathtext'] = True
params['axes.axisbelow'] = True
params['axes.labelpad'] = 1*F           #points
# Ticks
params['xtick.color'] = 'black'
params['ytick.color'] = 'black'
params['xtick.labelsize'] = 'small'
params['ytick.labelsize'] = 'small'
params['xtick.major.width'] = 1*F       #points
params['ytick.major.width'] = 1*F       #points
params['xtick.major.size'] = 2*F        #points
params['ytick.major.size'] = 2*F        #points
params['xtick.major.pad'] = 1*F         #points
params['ytick.major.pad'] = 1*F         #points
params['xtick.minor.width'] = 0.5*F     #points
params['ytick.minor.width'] = 0.5*F     #points
params['xtick.minor.size'] = 1*F        #points
params['ytick.minor.size'] = 1*F        #points
params['xtick.minor.pad'] = 1*F         #points
params['ytick.minor.pad'] = 1*F         #points
# Legend
params['legend.markerscale'] = 1        #relative to markersize/linewidth
params['legend.fontsize'] = 'small'
params['legend.labelspacing'] = 0.25    #fraction of font size
params['legend.frameon'] = False
params['legend.borderpad'] = 0.25       #fraction of font size
params['patch.linewidth'] = 0.25*F      #points
params['legend.fancybox'] = False
# params['legend.edgecolor'] = 'black'
# Saving
params['figure.figsize'] = (1.7*F, 1.1*F)   #inches
params['savefig.bbox'] = 'standard'
params['savefig.pad_inches'] = 0            #inches
params['savefig.dpi'] = 200
params['savefig.transparent'] = False
params['svg.fonttype'] = 'path'
# Imshow
params['image.aspect'] = 'auto'
params['image.origin'] = 'lower'

mpl.style.use(params)

class colors:
    # valence = np.array([ 27,120, 55]) / 255
    # social = np.array([ 90,174, 97]) / 255
    # somatic = np.array([153,112,171]) / 255
    # mood = np.array([118, 42,131]) / 255

    valence = 'slategrey'
    social = 'slategrey'
    somatic = 'slategrey'
    mood = 'slategrey'

    ac = '#01153E'
    mc = '#029386'
    energy = '#8C000F'

    standard = 'slategrey'
    lightgrey = 'whitesmoke'
    white = 'white'

    errorbars = 'black'

    binary_cmap = mpl.colors.ListedColormap([white, standard])

    features = list(reversed([valence, valence, social, 
                              social, social, social, 
                              somatic, somatic, somatic,
                              mood, mood, mood, mood, mood, mood]))