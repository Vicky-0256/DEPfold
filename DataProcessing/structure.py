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



