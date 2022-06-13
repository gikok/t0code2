import pandas as pd
import numpy as np
import re

def add_or(s):
    li = s.rsplit(', ')
    start = ', '.join(li[:-1])
    end =  ' or ' + li[-1]
    return start+end
    

def create_option_string(df, column, inds):
    options = ""
    for i in inds:
        options +=  f"'{df[column].iloc[i]}', "
        if i == inds[-1]:
            options = options[:-2]
    options = add_or(options)
        
    return options

def get_other_indices(index, size, num):
    """
    given a number {index}, returns {num} integers in range({size})
    """
    
    proceed = 0
    while proceed == 0:
        inds = np.random.choice(range(size), num)
        # check if any of the new indices is same as initial
        if (inds == index).any()==False:
            proceed = 1
        else:
            proceed = 0
    
    return inds


def no_to_name(index):
    
    # get random number of options
    num = np.random.randint(1, 4)
    
    # get random item index
    inds = get_other_indices(index, len(df), num)
    inds = np.append(inds, index)
    np.random.shuffle(inds)
    
    # get answer options
    options = create_option_string(df, 'name', inds)
    
    # create the input string
    inp = f"if item_no is {df['item_no'].iloc[index]}, which of the following is the correct name: " + options + '?'
    
    
    # set the target
    target = df['name'].iloc[inds[np.argmin(inds-index)]]
    
    return inp.lower(), target.lower()

def name_to_no(index):
    
    # get random number of options
    num = np.random.randint(1, 4)
    
    # get random item index
    inds = get_other_indices(index, len(df), num)
    inds = np.append(inds, index)
    np.random.shuffle(inds)
    
    # get answer options
    options = create_option_string(df, 'item_no', inds)
    
    # get the input string
    inp = f"if name is {df['name'].iloc[index]}, what item_no does it refer to? " + options + '?' 
    
    # get the target
    target = df['item_no'].iloc[inds[np.argmin(inds-index)]]
    
    return inp.lower(), target.lower()

def is_description(index):
    
    is_true = np.random.randint(2)
    
    if is_true:
        desc = df['benefits'].iloc[index]
        target = "yes"
    else:
        tempdf = df[df['benefits']!=df['benefits'].iloc[index]]
        desc = np.random.choice(tempdf['benefits'])
        target = "no"
        
    # make sure ends with period    
    desc = desc if desc.endswith(".") else desc + "."
    
    # create input
    inp = desc + f" is the previous sentence a description of item_no {df['item_no'].iloc[index]}. yes or no?"
    
    return inp.lower(), target

def is_summary(index):
    
    is_true = np.random.randint(2)
    
    if is_true:
        desc = df['key_w'].iloc[index]
        target = "yes"
    else:
        tempdf = df[df['key_w']!=df['key_w'].iloc[index]]
        desc = np.random.choice(tempdf['key_w'])
        target = "no"
        
    # make sure ends with period    
    desc = desc if desc.endswith(".") else desc + "."
    
    # create input
    inp = desc + f" is the previous sentence a summary of item_no {df['item_no'].iloc[index]}. yes or no?"
    
    return inp.lower(), target

def true_query(query_df, item_no):
    
    is_true = np.random.randint(2)
    
    if is_true:
        query = query_df['clean_query'].sample().iloc[0]
        target = "yes"
    else:
        query = query_df['clean_query'].sample().iloc[0]
        target = "no"

    
    # create input
    inp = "query:'" + query + f"'\ndoes the query above return item_no {item_no} as a result. yes or no?"
    
    return inp.lower(), target

def query_rank(query_df, item_no):

    # select random row
    row = query_df.sample()
    
    # get query string
    query = row['clean_query'].iloc[0]
    
    # get rank of item_no
    cols = np.array(row.columns)
    inds = (row[row==item_no].isnull()==False).values[0]
    
    # get first true value, that is not 0 (queries can be for item_no) and
    # sometimes same item_no has multiple ranks
    if len(inds)>1:
        inds[0] = False
        rank = cols[np.argmax(inds)][-1]
    
    # in case rank is 10
    rank = '10' if rank=='0' else rank
    
    # create possible options
    n_options = np.random.randint(1, 5)
    
    options = list(np.arange(1, 11))
    options.remove(int(rank))
    
    options = np.append(np.random.choice(options, n_options).astype(str), rank)
    np.random.shuffle(options)
    
    # create input
    inp = "query:'" + query + f"'\nthe query above returns item_no {item_no} as a result. what is its rank? " + add_or(', '.join(options))
    
    target = rank
    
    return inp.lower(), rank