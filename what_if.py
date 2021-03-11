from math import *
import pandas
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Select
from bokeh.palettes import Oranges5, Greens9, Greens256
from bokeh.plotting import figure
from sklearn.neighbors import NearestNeighbors
import numpy as np
from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Rect, Text
import numpy as np
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, HoverTool, TapTool
from bokeh.models.glyphs import Text
from bokeh.io import curdoc, show

# Importing the given data in csv file
df = pandas.read_csv('activitydata.csv')
# converting the activity sequences into lists from panda table
data = df['activity'].values.tolist()
patient_id = df['ID'].values.tolist()
segment = df['segment'].values.tolist()
# Converting the score column in the panda table into a list
DMscore = df['score'].values.tolist()
data = [s.strip('S') for s in data]
longest_string = max(data, key=len)
empty = []
for string in data:
    empty.append(string.zfill(len(longest_string)))
empty2 = []
for element in empty:
    string = ''
    for char in element:
        if char == '0':
            string += '1'
        if char == 'L':
            string += '2'
        if char == 'M':
            string += '3'
        if char == 'V':
            string += '4'
        if char == 'Y':
            string += '5'
    empty2.append([string])

empty3 = []
for e in empty2:
    empty3.append(list(''.join(e)))
DMseq = [[int(str(j)) for j in i] for i in empty3]


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def index_neighbours(seq, n_neighbors):
    input_array = NearestNeighbors(n_neighbors)
    reshape_to_2d = np.reshape(seq, (1, -1))
    input_array.fit(reshape_to_2d)
    input_array = input_array.kneighbors(reshape_to_2d, return_distance=False)
    return input_array


def average_score(seq, num_neighbors):
    average_list = []
    index = index_neighbours(seq, num_neighbors)
    for i in index:
        sub_scores = [DMscore[x] for x in i]
        avg = sum(sub_scores)/len(sub_scores)
        average_list.append(avg)
    return average_list

print(average_score(DMseq[10], 5))

stringList = []
for x in DMscore:
    stringList.append(str(x))

# first prediction
def score_cur(s,n):
    average = average_score(s, n)
    return average[s]


def score_up(seq, slot, s, n):
    if seq[slot-1] in [2, 3, 4]:
        seq[slot-1] += 1
        # new_score = score_cur(s,n)
    return seq


def score_down(seq, slot, s, n):
    if seq[slot-1] in [3, 4, 5]:
        seq[slot - 1] -= 1
        # new_score = score_cur(s, n)
    return seq


strings = [str(integer) for integer in segment]
PatSeqID = [m+"_"+str(n) for m, n in zip(patient_id, strings)]
dictionary = dict(zip(PatSeqID, DMseq))
dictionary2 = dict(zip(PatSeqID, stringList))
menu = PatSeqID
dropdown = Select(title='DMseq',value='BC_0083_0', options=menu)
seg_color = Oranges5
seg2_color = Greens9
seg3_color = Greens256
N_neighbours = 5  # number of neighbours used for KNN
# plot = figure(plot_width=900, plot_height=200, min_border=0, sizing_mode='scale_height', tools='pan,box_zoom,hover,reset')
plot = figure(title=None, title_location="right", plot_width=900, plot_height=200, min_border=0, toolbar_location="right")

def make_rectangles():
    N = 41
    x = np.linspace(0, 41, N)
    c = ['' for c in range(41)]
    u = ['' for u in range(41)]
    d = ['' for d in range(41)]
    # t = ['' for t in range(10)]

    title = dictionary2[dropdown.value][:3]

    seq = dictionary[dropdown.value]
    seq2 = dictionary2[dropdown.value]
    print(dictionary2)

    for i in range(41):
        e = seq[i]
        u[i] = seg2_color[9 - e]

    for i in range(41):
        e = seq[i]
        c[i] = seg_color[5 - e]

    for i in range(41):
        e = seq[i]
        d[i] = seg2_color[9 - e]

    # for i in range(10):
    #     t[i] = seq2[i]

    source = ColumnDataSource(dict(x=x, c=c))
    source_up = ColumnDataSource(dict(x=x, u=u))
    source_down = ColumnDataSource(dict(x=x, d=d))
    #source_text = ColumnDataSource(dict(x=x, t=t))

    plot = figure(plot_width=1000, plot_height=250, min_border=0, toolbar_location="right")
    plot.title.text = title
    plot.title.align = "right"
    plot.title.text_font_size = "25px"
    glyph = Rect(x="x", y=0,  width=1, height=10, fill_color="c")
    glyph_up = Rect(x="x", y=6, width=1, height=3, fill_color="u")
    glyph_down = Rect(x="x", y=-6, width=1, height=3, fill_color="d")
    glyph_text = Text(x=42, y=0)
    plot.add_glyph(source, glyph)
    plot.add_glyph(source_up, glyph_up)
    plot.add_glyph(source_down, glyph_down)
    #plot.add_glyph(source_text, glyph_text)
    plot.add_tools(HoverTool())
    plot.add_tools(TapTool())

    return plot


def update(attr, old, new):
    layout.children[1] = make_rectangles()


dropdown.on_change('value', update)
controls = column(dropdown)
layout = row(controls, make_rectangles())
curdoc().add_root(layout)
curdoc().title = "what_if"