import matplotlib as mpl

def mm2inch(*tupl):
    inch = 25.4 #1 inch = 25.4 mm
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# Width of the figure in mm (using Elsevier guidelines)
# 190 = full page image
# 140 = medium size image
# 90 = small image
# e.g. for a very large fullsize image use figsize=mm2inch(190, 1.2*190)

# example usage:
# figsize=mm2inch(190, 1.2*190)
# plt.Figure(figsize=figsize)
# plt.plot()
#
# or, with subplots:
# plt.subplots(figsize=figsize)

mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['figure.dpi'] = 500

mpl.rc('font',**{'family':'serif','serif':['Times']})
# mpl.rc('text', usetex=True) # use this if you want to use latex