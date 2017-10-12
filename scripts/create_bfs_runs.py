import os

EXE_PATH = '/home/jrc210/lonestar/bin/'

INPUT_DIR = '/home/jrc210/lonestar/inputs/'

OUTPUT_DIR = '/home/jrc210/lonestar/outputs/'

result_tail = '.res'

varients = ['bfs', 'bfs-atomic', 'bfs-wla', 'bfs-wlc', 'bfs-wlw']

for varient in varients:
    #if there is a directory for that day,
    if os.path.exists(INPUT_DIR):

        #create the output directory if it does not exit
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        #get the files in the input directory
        fileList = os.listdir(INPUT_DIR)

        #create the shell script commands
        for fileName in fileList:
            if fileName[0] != '.':  #make sure the file is not a hidden file
                if '.gr' in fileName and not '.sym.' in fileName: #test for appropriate input file
                    print "echo 'processing %s with input %s' " %(varient, fileName)
                    print '%s%s %s%s > %s%s%s ' %(EXE_PATH, varient, INPUT_DIR, fileName, OUTPUT_DIR, fileName, result_tail)
