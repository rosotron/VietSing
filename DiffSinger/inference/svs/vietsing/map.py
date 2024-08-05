def viet_word2ph_func():
    word2phs = {'sp': 'sp', 'sil': 'sil', 'spn':'spn'}
    with open('inference/svs/viet/viet_word2ph.txt') as rf:
        for line in rf.readlines():
            elements = [x.strip() for x in line.split('\t') if x.strip() != '']
            word2phs[elements[0]] = elements[1]
    return word2phs