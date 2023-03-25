
'''
author: Roberto Scalas 
date:   2023-03-24 13:33:15.438225
'''
import numpy as np
import streamlit as st
st.set_page_config(page_title='Cellular Automata', layout = 'wide', initial_sidebar_state = 'auto')
import plotly.express as px
import plotly.graph_objects as go

st.sidebar.title('Cellular Automata')

size_array = st.sidebar.slider('Size array', 1, 1000, 180)
epochs = st.sidebar.slider('Epochs', 1, 1000, 250)
rule_n = st.select_slider('Select the rule', options = list(range(256)), value = 156)

with st.expander('Customize Array', expanded = False):
    # row is an nnp array of 0 size 10, 1
    row = np.zeros(size_array) 
    selectors = st.multiselect('Select the initial state', list(range(size_array)), default = [int(size_array/2)] ) # set mmiddle value to 1
    st.subheader('Initial state: Gen 0')

    for i in selectors:
        row[i] = 1

    # plot as heatmap
    row_ = row.reshape(1, size_array)
    fig = px.imshow(row_)
    fig.update_xaxes(title_text=f'Length of the array: {size_array}')
    fig.update_yaxes(title_text=f'Number of epochs: {epochs}')
    fig.update_traces(zmin=0, zmax=1)
    fig.update_layout(xaxis=dict(side="top"))
    fig.update_layout(xaxis=dict(tickmode='linear'))
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    # no axis
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)


    st.plotly_chart(fig, use_container_width=True)

def get_all_permutations(array):
    '''
    This function get all the possible permutaion of of an array of 0 and 1
    '''
    # calculate all possible configuration of a n bit number
    # get all the possible permutaion of of an array of 0 and 1
    permutations = []
    for i in range(2**len(array)):
        permutations.append([int(x) for x in np.binary_repr(i, width=len(array))])
    return permutations

# get all the possible permutaion of of an array of 0 and 1 of size 8
permutations = get_all_permutations([0, 0, 0, 0, 0, 0, 0, 0]) # sort them in the same order of the rule 30
rules_results = permutations[rule_n]

triplets = []
for i in range(len(row)):
    if i == 0:
        triplets.append([0, row[i], row[i+1]])
    elif i == len(row) - 1:
        triplets.append([row[i-1], row[i], 0])
    else:
        triplets.append([row[i-1], row[i], row[i+1]])
        
# unique triplets
unique_triplets = []
for triplet in triplets:
    if triplet not in unique_triplets:
        unique_triplets.append(triplet)


with st.expander('Triplets Rule: {}'.format(rule_n)):
    # find all possible permutations of the triplets
    unique_triplets = get_all_permutations([0, 0, 0])
    columns = st.columns(len(unique_triplets))
    for t in range(len(unique_triplets)):
        fig = px.imshow(np.array(unique_triplets[t]).reshape(1, 3), range_color=[0, 1])
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(zmin=0, zmax=1, showscale=False)
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        # set colors if 0 is white and 1 is black
        columns[t].plotly_chart(fig, use_container_width=True)
        columns[t].subheader(f'{rules_results[t]}')
    
def rule(row, rule_number = 30):
    '''
    This function divide the row in triplets and apply the rule 30
    '''
    rules_results = permutations[rule_number]
    triplets = []
    for i in range(len(row)):
        if i == 0:
            triplets.append([0, row[i], row[i+1]])
        elif i == len(row) - 1:
            triplets.append([row[i-1], row[i], 0])
        else:
            triplets.append([row[i-1], row[i], row[i+1]])
    #print(triplets)
    new_grid = []
    for triplet in triplets:
        if triplet == [0, 0, 0]:
            new_grid.append(rules_results[0])
        elif triplet == [0, 0, 1]:
            new_grid.append(rules_results[1])
        elif triplet == [0, 1, 0]:
            new_grid.append(rules_results[2])
        elif triplet == [0, 1, 1]:
            new_grid.append(rules_results[3])
        elif triplet == [1, 0, 0]:
            new_grid.append(rules_results[4])
        elif triplet == [1, 0, 1]:
            new_grid.append(rules_results[5])
        elif triplet == [1, 1, 0]:
            new_grid.append(rules_results[6])
        elif triplet == [1, 1, 1]:
            new_grid.append(rules_results[7])
    print(new_grid)
    return new_grid

import pandas as pd

def stats_nerd(row):
    # get sum of 1s 
    sum_1s = sum(row)
    # get sum of 0s
    sum_0s = len(row) - sum_1s
    # get the percentage of 1s
    perc_1s = sum_1s / len(row)
    # get the percentage of 0s
    perc_0s = sum_0s / len(row)
    return sum_1s, sum_0s, perc_1s, perc_0s

data = []
data_frame_stats = pd.DataFrame(columns=['number of 1', 'number of 0', '% single generation (1)', '% single generation (0)'])
for i in range(epochs):
    #row = rule_30(row)
    row = rule(row, rule_number =rule_n)
    # get stats
    sum_1s, sum_0s, perc_1s, perc_0s = stats_nerd(row)
    data_frame_stats.loc[i] = [sum_1s, sum_0s, perc_1s, perc_0s]
    data.append(row)

# add the percentage of 1s and 0s
data_frame_stats['% over generations (1)'] = 0
data_frame_stats['% over generations (0)'] = 0
for i in range(epochs):
    # get that [0:i] rows
    data_ = data[0:i+1]
    # calculate the percentage of 1s
    data_frame_stats['% over generations (1)'][i] = sum(data_frame_stats['% single generation (1)'][0:i+1]) / (i+1)
    # calculate the percentage of 0s
    data_frame_stats['% over generations (0)'][i] = sum(data_frame_stats['% single generation (0)'][0:i+1]) / (i+1)

df = pd.DataFrame(data)
c1,c2 = st.columns(2)
c1.dataframe(df)

# plot as plotly
import plotly.express as px
fig = px.imshow(df)
# set x axis title as length of the array
fig.update_xaxes(title_text=f'Length of the array: {size_array}')
# set y axis title as number of epochs
fig.update_yaxes(title_text=f'Number of epochs: {epochs}')
# no colorbar
fig.update_layout(coloraxis_showscale=False)
# no axis, mke it bigger
fig.update_layout(showlegend=False, coloraxis_showscale=False)
c2.plotly_chart(fig, use_container_width=True)

# stats
st.subheader('Stats')
with st.expander('Show stats table'):
    st.dataframe(data_frame_stats)


# plot as time series
fig = go.Figure()
# plot percentage of 1s
fig.add_trace(go.Scatter(x=data_frame_stats.index, y=data_frame_stats['% single generation (1)'], name='% single generation (1)', mode='lines+markers'))
# plot percentage of 0s
fig.add_trace(go.Scatter(x=data_frame_stats.index, y=data_frame_stats['% single generation (0)'], name='% single generation (0)', mode='lines+markers'))
# plot percentage of 1s over generations
fig.add_trace(go.Scatter(x=data_frame_stats.index, y=data_frame_stats['% over generations (1)'], name='% over generations (1)', mode='lines+markers'))
# plot percentage of 0s over generations
fig.add_trace(go.Scatter(x=data_frame_stats.index, y=data_frame_stats['% over generations (0)'], name='% over generations (0)', mode='lines+markers'))
# add line on mean
fig.update_layout(title_text='Percentage of 1s and 0s')
st.plotly_chart(fig, use_container_width=True)