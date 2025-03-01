def load_ct(ctFn, load_all=False):
    """
    copy from IPyRSSA
    
    Read ct file
    
    ctFn                -- ct file name
    load_all            -- load all ct from ct file, or load the first one
    
    Return:
        [seq,dotList,length] if load_all==False
        {1:[seq,dotList,length], ...} if load_all==True
    """
    Ct = {}
    
    ID = 1
    ctList = []
    seq = ""
    last_id = 0
    
    seqLen = 0
    headline = ""
    
    for line in open(ctFn):
        line = line.strip()
        if line[0]=='#':
            continue
        data = line.strip().split()
        if not data[0].isdigit():
            raise RuntimeError("cf file format Error: the first item should be a digit")
        elif seqLen==0:
            seqLen = int(data[0])
            headline = line.strip()
        elif int(data[0])!=last_id+1:
            raise RuntimeError("ct file format error...")
        else:
            left_id = int(data[0])
            right_id = int(data[4])
            seq += data[1]
            if right_id != 0 and left_id<right_id:
                ctList.append((left_id, right_id))
            last_id += 1
            if left_id == seqLen:
                #print(data, last_id+1)
                Ct[ID] = [seq, ctList, seqLen, headline]
                assert seqLen==len(seq)
                last_id = 0
                seq = ""
                ctList = []
                ID += 1
                seqLen = 0
                if not load_all:
                    return Ct[1]
    
    if seq:
        Ct[ID] = [seq, ctList, seqLen]
        if seqLen != left_id:
            raise RuntimeError("ct file format error...")
    
    return Ct

# def load_ct(ctFn, load_all=False):
#     """
#     Read ct file
    
#     ctFn                -- ct file name
#     load_all            -- load all ct from ct file, or load the first one
    
#     Return:
#         [seq,dotList,length] if load_all==False
#         {1:[seq,dotList,length], ...} if load_all==True
#     """
#     Ct = {}
    
#     ID = 1
#     ctList = []
#     seq = ""
#     last_id = 0
    
#     seqLen = 0
#     headline = ""
    
#     try:
#         with open(ctFn, 'r') as f:
#             for line_num, line in enumerate(f, 1):
#                 line = line.strip()
#                 if line.startswith('#'):
#                     continue
#                 data = line.split()
#                 if not data:  # Skip empty lines
#                     continue
#                 if not data[0].isdigit():
#                     if seqLen == 0:  # This might be the header line
#                         try:
#                             seqLen = int(data[0])
#                             headline = line
#                             continue
#                         except ValueError:
#                             raise RuntimeError(f"CT file format Error at line {line_num}: the first item should be a digit")
#                     else:
#                         raise RuntimeError(f"CT file format Error at line {line_num}: the first item should be a digit")
                
#                 left_id = int(data[0])
                
#                 if seqLen == 0:
#                     seqLen = left_id
#                     headline = line
#                 elif left_id != last_id + 1:
#                     raise RuntimeError(f"CT file format error at line {line_num}: ID not continuous")
#                 else:
#                     right_id = int(data[4])
#                     seq += data[1]
#                     if right_id != 0 and left_id < right_id:
#                         ctList.append((left_id, right_id))
#                     last_id = left_id
#                     if left_id == seqLen:
#                         Ct[ID] = [seq, ctList, seqLen, headline]
#                         assert seqLen == len(seq), f"Sequence length mismatch: expected {seqLen}, got {len(seq)}"
#                         last_id = 0
#                         seq = ""
#                         ctList = []
#                         ID += 1
#                         seqLen = 0
#                         if not load_all:
#                             return Ct[1]
        
#         if seq:
#             Ct[ID] = [seq, ctList, seqLen, headline]
#             if seqLen != last_id:
#                 raise RuntimeError("CT file format error: Unexpected end of file")
        
#         return Ct if load_all else Ct[1]
    
#     except Exception as e:
#         print(f"Error processing file {ctFn}: {str(e)}")
#         return None  # Return None to indicate that this file had an error



def parse_pseudoknot(ctList):
    """
    copy from IPyRSSA

    ctList              -- paired-bases: [(3, 8), (4, 7)]
    
    Parse pseusoknots from clList
    Return:
        [ [(3, 8), (4, 7)], [(3, 8), (4, 7)], ... ]
    """
    ctList.sort(key=lambda x:x[0])
    ctList = [ it for it in ctList if it[0]<it[1] ]
    paired_bases = set()
    for lb,rb in ctList:
        paired_bases.add(lb)
        paired_bases.add(rb)
    
    # Collect duplex
    duplex = []
    cur_duplex = [ ctList[0] ]
    for i in range(1, len(ctList)):
        bulge_paired = False
        for li in range(ctList[i-1][0]+1, ctList[i][0]):
            if li in paired_bases:
                bulge_paired = True
                break
        if ctList[i][1]+1>ctList[i-1][1]:
            bulge_paired = True
        else:
            for ri in range(ctList[i][1]+1, ctList[i-1][1]):
                if ri in paired_bases:
                    bulge_paired = True
                    break
        if bulge_paired:
            duplex.append(cur_duplex)
            cur_duplex = [ ctList[i] ]
        else:
            cur_duplex.append(ctList[i])
    if cur_duplex:
        duplex.append(cur_duplex)
    
    # Discriminate duplex are pseudoknot
    Len = len(duplex)
    incompatible_duplex = []
    for i in range(Len):
        for j in range(i+1, Len):
            bp1 = duplex[i][0]
            bp2 = duplex[j][0]
            if bp1[0]<bp2[0]<bp1[1]<bp2[1] or bp2[0]<bp1[0]<bp2[1]<bp1[1]:
                incompatible_duplex.append((i, j))
    
    pseudo_found = []
    while incompatible_duplex:
        # count pseudo
        count = {}
        for l,r in incompatible_duplex:
            count[l] = count.get(l,0)+1
            count[r] = count.get(r,0)+1
        
        # find most possible pseudo
        count = list(count.items())
        count.sort( key=lambda x: (x[1],-len(duplex[x[0]])) )
        possible_pseudo = count[-1][0]
        pseudo_found.append(possible_pseudo)
        i = 0
        while i<len(incompatible_duplex):
            l,r = incompatible_duplex[i]
            if possible_pseudo in (l,r):
                del incompatible_duplex[i]
            else:
                i += 1
    
    pseudo_duplex = []
    for i in pseudo_found:
        pseudo_duplex.append(duplex[i])
    
    return pseudo_duplex

def remove_pseudoknot_pairs(ctList, pseudo_duplex):
    """
    ctList              -- paired-bases: [(3, 8), (4, 7)]
    pseudo_duplex       -- pseudoknots: [(3, 8)]

    Remove pseudoknot pairs from ctList
    Return normal_pairs:
        [(4, 7)]
    """
    # 将假结碱基对放入集合中
    pseudo_set = set()
    for duplex in pseudo_duplex:
        for pair in duplex:
            pseudo_set.add(pair)
    
    # 移除假结碱基对，保留普通碱基对
    normal_pairs = [pair for pair in ctList if pair not in pseudo_set]
    
    return normal_pairs

def ct2pair(ctList):
    """
    ctList              -- paired-bases: [(3, 8), (4, 7)]
    
    Convert ctlist structure to normal pair and pseudo pairs

    Return:
        normal_pairs -- List of normal pairs
        pseudo_pairs -- List of pseudo pairs
    """
    if not ctList:
        return [], []
    
    # Sort and filter ctList
    ctList = sorted(ctList, key=lambda x: x[0])
    ctList = [it for it in ctList if it[0] < it[1]]
    
    # Parse pseudoknots and remove them from the normal pairs
    pseudo_pairs = parse_pseudoknot(ctList)
    normal_pairs = remove_pseudoknot_pairs(ctList, pseudo_pairs)

    return normal_pairs, pseudo_pairs


def ct2dot_and_pairs(ctList, length):
    """
    ctList              -- paired-bases: [(3, 8), (4, 7)]
    length              -- Length of structure
    
    Convert normal_pairs structure to dot-bracket and return normal pairs and pseudo pairs
    [(3, 8), (4, 7)]  => ..((..))..

    Return:
        dot_bracket  -- Dot-bracket notation string
        normal_pairs -- List of normal pairs
        pseudo_pairs -- List of pseudo pairs
    """

    dot = ['.'] * length
    if not ctList:
        return "".join(dot), [], []

    # Sort and filter ctList
    ctList = sorted(ctList, key=lambda x: x[0])
    ctList = [it for it in ctList if it[0] < it[1]]
    
    # Parse pseudoknots and remove them from the normal pairs
    pseudo_pairs = parse_pseudoknot(ctList)
    normal_pairs = remove_pseudoknot_pairs(ctList, pseudo_pairs)
    
    # Mark normal pairs in dot-bracket notation
    for l, r in normal_pairs:
        dot[l - 1] = '('
        dot[r - 1] = ')'
    
    return "".join(dot), normal_pairs, pseudo_pairs


def ct2dot(ctList, length):
    """
    ctList              -- paired-bases: [(3, 8), (4, 7)]
    length              -- Length of structure
    
    Convert ctlist structure to dot-bracket
    [(3, 8), (4, 7)]  => ..((..))..
    """
    dot = ['.']*length
    if len(ctList) == 0:
        return "".join(dot)
    ctList = sorted(ctList, key=lambda x:x[0])
    ctList = [ it for it in ctList if it[0]<it[1] ]
    if len(ctList) > 0:

        pseudo_duplex = parse_pseudoknot(ctList)
        for l,r in ctList:
            dot[l-1] = '('
            dot[r-1] = ')'
        dottypes = [ '<>', r'{}', '[]' ]
        if len(pseudo_duplex)>len(dottypes):
            print("Warning: too many psudoknot type: %s>%s" % (len(pseudo_duplex),len(dottypes)))
        for i,duplex in enumerate(pseudo_duplex):
            for l,r in duplex:
                dot[l-1] = dottypes[i%3][0]
                dot[r-1] = dottypes[i%3][1]
    return "".join(dot)

