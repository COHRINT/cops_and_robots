import matplotlib.pyplot as pyplot

def writePBM(src,name='pbm_gen'):
    outFile = open(name+'.pbm', 'w')
    outFile.write("P1\n")
    outFile.write(str(src.shape[0]) + ' ' + str(src.shape[1]) + '\n')
    for i in range(0, src.shape[0]):
        for j in range(0, src.shape[1]):
            outFile.write(str(int(src[i,j])) + ' ')
        outFile.write('\n')

    outFile.close()
    return
