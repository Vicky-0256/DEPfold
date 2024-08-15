def is_connect(a,b):
    if a == '(':
        if b == '(':
            # keep
            return 2,1
        if b == ')':
            return 0,0
        if b == 1:
            return 0,-1
    if a == 0:
        if b == ')':
            return 1,0
        if b == 0:
            return 0,-1
        if b == '(':
            return 2,1


def get_tree(structure):
    dot_indices = [index for index, value in enumerate(structure) if value == '.']
    brac_indices = [index for index, value in enumerate(structure) if value in ('(', ')')]
    # print(brac_indices)

    if len(brac_indices) == 0:
        parse_pair = []
    else:
        parse_pair = get_pair([structure[i] for i in brac_indices], brac_indices)

    result_list = []
    temp_list = []

    for i, dot_index in enumerate(dot_indices):
        if i == 0 or dot_index == dot_indices[i-1] + 1:
            temp_list.append(dot_index)
        else:
            result_list.append(temp_list)
            temp_list = [dot_index]

    result_list.append(temp_list)  # Append the last sublist

    # print(result_list)

    if len(brac_indices) == 0:
        last_node = 0
    else:
        last_node = brac_indices[-1]

    pair_list = get_pairs(result_list, last_node)

    return pair_list, parse_pair

def get_pair(seq,idx):

    i = 0
    new_node_idx = len(seq)
    parse_pair = []

    while i != -1 & len(seq) != 1:

        index = i

        if seq[index] == 1:
            a = seq[index-1]
            b = seq[index]
            preindex = index-1
            change_idx = index
        elif seq[index] == 0:
            if seq[index-1] == 0:
                a = seq[index-1]
                b = seq[index]
                change_idx = index
                preindex = index-1
            else:
                a = seq[index]
                b = seq[index+1]
                preindex = index
                change_idx = index+1
        else:
            a = seq[index]
            b = seq[index+1]
            change_idx = index+1
            preindex = index
       

        c,d = is_connect(a,b)
        if c != 2:

            parse_pair.append((idx[change_idx],idx[preindex]))
    

            seq[change_idx]=c
            
            idx.pop(change_idx-1)
            seq.pop(change_idx-1)
            
            new_node_idx += 1 

        
        i += d
        
    return parse_pair

def get_pairs(lst, last_node):
    result = []

    # 如果lst为空或其子列表为空, 直接返回空的result
    if not lst or all(not sub for sub in lst):
        return result

    for sub_lst in lst:
        # 跳过空的子列表
        if not sub_lst:
            continue

        if sub_lst is lst[-1] and last_node < lst[-1][0]:
            result.extend((sub_lst[i], sub_lst[i]-1) for i in range(len(sub_lst)))
        else:
            result.extend((sub_lst[i], sub_lst[i-1]) for i in range(1, len(sub_lst)))

            if sub_lst[-1] < last_node:
                result.append((sub_lst[-1] + 1, sub_lst[-1]))

    return result