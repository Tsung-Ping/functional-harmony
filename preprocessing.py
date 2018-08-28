import numpy as np
import math
import xlrd
import numpy.lib.recfunctions as rfn
from scipy import stats
import itertools
import os

def strided_axis1(a, window, hop):

    # Zero-padding
    npad = (a.shape[1] - window) % hop + 1
    if npad != 0 and hop != 1:
        b = np.lib.pad(a, ((0, 0), (0,npad)), 'constant', constant_values=0)
    else:
        b = np.array(a)

    # Length of 3D output array along its axis=1
    nd1 = int((b.shape[1] - window)/hop) + 1

    # Store shape and strides info
    m, n = b.shape
    s0, s1 = b.strides

    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(b, shape=(nd1, m, window), strides=(s1*hop, s0, s1))

def load_notes(directory="\\BPS_FH_Dataset\\", resolution=8):
    """
    Load notes in each piece, which is then represented as pianoroll.
    :param directory: the path of the dataset
    :param resolution: time resolution, default =  8 frame for one crotchet beat (that is, 32th note as 1unit in piano roll)
    :return: pieces, tdeviation
    """

    dt = [('onset', 'float'), ('pitch', 'int'), ('mPitch', 'int'), ('duration', 'float'), ('staffNum', 'int'), ('measure', 'int')] # datatype
    pieces = [None for _ in range(32)]
    tdeviation = [None for _ in range(32)] # time deviation
    for i in range(32):
        fileDir = directory + str(i+1).zfill(2) + "\\notes.csv"
        notes = np.genfromtxt(fileDir, delimiter=',', dtype=dt) # read notes from .csv file
        length = math.ceil((max(notes[-20:]['onset'] + notes[-20:]['duration']) - notes[0]['onset'])*resolution) # length of the piece
        tdeviation[i] = abs(notes[0]['onset']) # deviation of start time
        pianoRoll = np.zeros(shape=[128, length], dtype=np.int32)
        for note in notes:
            pitch = note['pitch']
            start = int(round((note['onset'] + tdeviation[i])*resolution))
            end = int(round((note['onset'] + note['duration'] + tdeviation[i])*resolution))
            time = range(start, end)
            pianoRoll[pitch,time] = 1 # add note to pianoRoll
        pieces[i] = pianoRoll

    return pieces, tdeviation

def load_chord_labels(directory="\\BPS_FH_Dataset\\"):
    """
    Load chords of each piece and add chord symbols into the labels.
    :param directory: the path of the dataset
    :return: chord_labels
    """

    dt = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U10')] # datatype
    chord_labels = [None for _ in range(32)]
    for i in range(32):
        fileDir = directory + str(i+1).zfill(2) + "\\chords.xlsx"

        workbook = xlrd.open_workbook(fileDir)
        sheet = workbook.sheet_by_index(0)
        chords = []
        for rowx in range(sheet.nrows):
            cols = sheet.row_values(rowx)
            if isinstance(cols[3], float): # if type(degree) == float
                cols[3] = int(cols[3])
            chords.append(tuple(cols))
        chords = np.array(chords, dtype=dt) # convert to structured array
        chord_labels[i] = r2tconvert(chords) # translate rchords into chord symbols

    return chord_labels

def r2tconvert(chords):
    """
    Translate roman numeral representations  into chord symbols, and add chord symbols into chord labels.
    :param chords:
    :return: rtchords
    """

    # Create scales of all keys
    temp = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    keys = {}
    for i in range(11):
        majtonic = temp[(i*4)%7] + int(i/7)*'+' + int(i%7>5)*'+'
        mintonic = temp[(i*4-2)%7].lower() + int(i/7)*'+' + int(i%7>2)*'+'

        scale = list(temp)
        for j in range(i):
            scale[(j+1)*4%7-1] += '+'
        majscale = scale[(i*4)%7:] + scale[:(i*4)%7]
        minscale = scale[(i*4+5)%7:] + scale[:(i*4+5)%7]
        minscale[6] += '+'
        keys[majtonic] = majscale
        keys[mintonic] = minscale

    for i in range(1, 9):
        majtonic = temp[(i*3)%7] + int(i/7)*'-' + int(i%7>1)*'-'
        mintonic = temp[(i*3-2)%7].lower() + int(i/7)*'-' + int(i%7>4)*'-'

        scale = list(temp)
        for j in range(i):
            scale[(j+2)*3%7] += '-'
        majscale = scale[(i*3)%7:] + scale[:(i*3)%7]
        minscale = scale[(i*3+5)%7:] + scale[:(i*3+5)%7]
        if len(minscale[6]) == 1:
            minscale[6] += '+'
        else:
            minscale[6] = minscale[6][:-1]

        keys[majtonic] = majscale
        keys[mintonic] = minscale

    # Translate chords
    outputQ = {'M':'M', 'm':'m', 'M7':'M7', 'm7':'m7', 'D7':'7', 'a':'aug', 'd':'dim', 'd7':'dim7', 'h7':'m7(b5)','a6':'7'}
    tchords = []
    for rchord in chords:
        key = str(rchord['key'])
        str_degree = str(rchord['degree'])

        if '/' not in str_degree: # case: not secondary chord
            if len(str_degree) == 1: # case: degree = x
                degree = int(float(str_degree))
                root = keys[key][degree-1]
            elif len(str_degree) == 2 and ('+' in str_degree[0] or '-' in str_degree[0]): # case: degree = -x or +x
                if str(rchord['quality']) != 'a6': # case: chromatic chord, -x
                    degree = int(float(str_degree[1]))
                    root = keys[key][abs(degree)-1]
                    if '+' not in root:
                        root += str_degree[0]
                    else:
                        root = root[:-1]
                else: # case: augmented 6th
                    degree = 6
                    root = keys[key][degree-1]
                    if str(rchord['key'])[0].isupper(): # case: major key
                        if '+' not in root:
                            root += '-'
                        else:
                            root = root[:-1]
            elif len(str_degree) == 2 and ('+' in str_degree[1] or '-' in str_degree[1]): # case: degree = x+
                degree = int(float(str_degree[0]))
                root = keys[key][degree - 1]

        elif '/' in str_degree: # case: secondary chord
            degree = str_degree
            if '+' not in degree.split('/')[0]:
                n = int(degree.split('/')[0]) # numerator
            else:
                n = 6
            d = int(degree.split('/')[1]) # denominator
            if d > 0:
                key2 = keys[key][d-1] # secondary key
            else:
                key2 = keys[key][abs(d)-1] # secondary key
                if '+' not in key2:
                    key2 += '-'
                else:
                    key2 = key2[:-1]

            if '+' in degree.split('/')[0]:
                n = 6
            root = keys[key2][n-1]
            if '+' in degree.split('/')[0]:
                if key2.isupper(): # case: major key
                    if '+' not in root:
                        root += '-'
                    else:
                        root = root[:-1]

        # Re-translate root for enharmonic equivalence
        if '++' in root: # if root = x++
            root = temp[(temp.index(root[0]) + 1)%7]
        elif '--' in root: # if root = x--
            root = temp[(temp.index(root[0]) - 1) % 7]

        if '-' in root: # case: root = x-
            if ('F' not in root) and ('C' not in root): # case: root = x-, and x != F and C
                root = temp[((temp.index(root[0]))-1)%7] + '+'
            else:
                root = temp[((temp.index(root[0]))-1)%7] # case: root = x-, and x == F or C
        elif ('+' in root) and ('E' in root or 'B' in root): # case: root = x+, and x == E or B
            root = temp[((temp.index(root[0]))+1)%7]

        quality = outputQ[str(rchord['quality'])]
        tchord = root + quality
        tchords.append(tchord)

    tchords = np.array(tchords, dtype= [('tchord', '<U10')])
    rtchords = rfn.merge_arrays((chords,tchords), flatten=True, usemask=False) # merge rchords and tchords into one structured array

    return rtchords

def segment_pianorolls(pianorolls, tdeviation, chord_labels, resolution=8, wsize=32, hsize=4, label_type=None):
    """
    Segment each pianoroll.
    :param pianorolls:
    :param tdeviation:
    :param chord_labels:
    :param resolution: time resolution,default=8 (32th note as 1unit in pianoroll)
    :param wsize: window size,  default= 32 (4 beats)
    :param hsize: hop size, default = 4 (half a beat)
    :param label_type: string, 'chord_symbol' and 'chord_function'  are valid
    :return:
    """

    segments_pianoroll = [[None for _ in range(32)] for _ in range(12)] # segments for 32 pieces, 12 keys
    segments_label = [[None for _ in range(32)] for _ in range(12)] # segment labels for 32 pieces, 12 keys
    for p in range(32):
        # Split Piano Roll into segments
        pianoroll = pianorolls[p]
        # Data augmentation
        pianoroll_aug = augment_pianoroll(pianoroll) # modulate to 12 keys
        for m in range(len(pianoroll_aug)):
            segments = strided_axis1(pianoroll_aug[m], window=wsize, hop=hsize).astype(np.float32) # Split pianoroll into segments with shape = [num_segments, 61, wsize]
            segments_reshape = [segments[i, :, :].reshape(61 * wsize) for i in range(segments.shape[0])] # reshape segments, new shape = [num_segments, 61*wsize]

            # Normalize each segment
            for i, v in enumerate(segments_reshape):
                if np.std(v) != 0:
                    segments_reshape[i] = stats.zscore(v)
            segments_pianoroll[m][p] = segments_reshape

        # Get corresponding chord label (only chord symbol) for each segment
        labels = []
        td = tdeviation[p] # time deviation
        num_segments = segments.shape[0] # number of segments
        for n in range(num_segments):
            sonset = n*hsize*(1/resolution) - td # onset time of the segment
            send = sonset + wsize*(1/resolution) # end time of the segment
            scenter = max(0, (sonset + send)/2)
            # print(fonset, fend, fcenter)
            label = chord_labels[p][(chord_labels[p]['onset'] <= scenter) & (chord_labels[p]['end'] > scenter)]
            if len(label) != 0:
                label = label[0]
            else:
                print('Error: Cannot read label!')
                print('piece %d' % p)
                print(label)
                quit()
            labels.append(label)

        if label_type == 'chord_symbol':
            labels_onehot = tchord2onehot(labels) # convert labels to one hot vectors
            labels_aug = augment_tchords(labels_onehot)
        elif label_type == 'chord_function':
            labels_onehot = rchord2onehot(labels)  # convert labels to one hot vectors
            labels_aug = augment_rchords(labels_onehot)
        else:
            print('LabelTypeError: %s,' % label_type, 'label_type should be \'chord_symbol\' or \'chord_function\'.')
            quit()
        for m in range(len(labels_aug)):
            segments_label[m][p] = labels_aug[m]

    return segments_pianoroll, segments_label

def augment_pianoroll(pianoroll):
    """
    Modulate pianoroll into 12 keys.
    :param pianoroll:
    :return: pianoroll_aug
    """

    pianoroll_aug = [None for _ in range(12)]
    for m in range(12):
        # Modulate to 12 keys
        if m < 7:
            temp = np.roll(pianoroll, shift=m, axis=0) # modulate upwards
        else:
            temp = np.roll(pianoroll, shift=m-12, axis=0) # modulate downwards

        # Cut off pitches higher than 84 (which is then moved a octave down)
        while (np.any(np.any(temp[85:], axis=1))):
            cut_off = np.array(temp[85:])
            temp[85:] -= cut_off
            temp[73:116] += cut_off

        # Cut off pitches lower than 24 (which is then moved a octave up)
        while (np.any(np.any(temp[:24], axis=1))):
            cut_off = np.array(temp[:24])
            temp[:24] -= cut_off
            temp[12:36] += cut_off

        pianoroll_aug[m] = temp[24:85] # only uses the pitch range from C1 (24) to C6 (84)

    return pianoroll_aug

def tchord2onehot(labels):
    """
    Convert chord symbols into one-hot vectors
    :param labels:
    :return: onehots
    """

    root_template = ['C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B']
    onehots = []
    for label in labels:
        # print('label =', label)
        tchord = label['tchord']
        # quit()
        onehot = [0 for i in range(25)]
        if '+' not in tchord and '-' not in tchord:
            root = tchord[0]
            quality = tchord[1:]
        else:
            root = tchord[:2]
            quality = tchord[2:]

        chord_hot = root_template.index(root)
        if quality in ['M', 'm', 'M7', 'm7', '7']:
            if quality == 'm' or quality == 'm7':
                chord_hot += 12
        else:
            chord_hot = 24

        onehot[chord_hot] = 1
        onehots.append(onehot)

    return onehots

def rchord2onehot(chords):

    # Translate chords to onehot vectors
    tonic_template = ['C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B']
    tonic_translation_dict = {'C-':'B', 'D-':'C+', 'E-':'D+', 'E+':'F', 'F-':'E', 'G-':'F+', 'A-':'G+', 'B-':'A+', 'B+':'C'}
    quality_template = ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6']
    one_hot_vectors = []
    for chord in chords:

        # Get attributes in chord labels
        key = str(chord['key'])
        degree = str(chord['degree'])
        quality = str(chord['quality'])
        inversoin = int(chord['inversion'])

        # Translate key to one-hot vector
        key_vector = [0 for _ in range(24)] # 24 major and minor modes, 0-11 for major keys, 12-23 for minor keys
        tonic = key.capitalize()
        if tonic in tonic_translation_dict.keys():
            tonic = tonic_translation_dict[tonic]
        tonic_hot = tonic_template.index(tonic)
        # check mode
        if key[0].islower():
            tonic_hot += 12
        key_vector[tonic_hot] = 1

        # Translate degree to one-hot vector
        degree_numerator_vector = [0 for _ in range(21)] # (7 diatonics *  3 chromatics  = 21: {0-6 diatonic, 7-13 sharp, 14-20 flat})
        degree_denominator_vector = [0 for _ in range(21)] # (7 diatonics *  3 chromatics  = 21: {0-6 diatonic, 7-13 sharp, 14-20 flat})
        # check numerator and denominator of degree
        if '/' not in degree:
            denominator = 1
            numerator = translate_degree(degree)
        else:
            numarator_str = degree.split('/')[0]
            denominator_str = degree.split('/')[1]
            numerator = translate_degree(numarator_str)
            denominator = translate_degree(denominator_str)
        degree_numerator_vector[numerator-1], degree_denominator_vector[denominator-1] = 1, 1

        # Translate quality to one-hot vector
        quality_vector = [0 for _ in range(10)] # {'M': 0, 'm': 1, 'd': 2, 'a': 3, 'M7': 4, 'm7': 5, 'D7': 6, 'd7': 7, 'h7': 8, 'a6': 9}
        quality_hot = quality_template.index(quality)
        quality_vector[quality_hot] = 1

        # Translate inversion to one-hot vector
        inversoin_vector = [0 for _ in range(4)] # {'ori.':0, '1st':1, '2nd', 2, '3rd': 3}
        inversoin_hot = inversoin
        inversoin_vector[inversoin_hot] = 1

        all_vectors = (key_vector,
                       degree_denominator_vector,
                       degree_numerator_vector,
                       quality_vector,
                       inversoin_vector)

        one_hot_vectors.append(all_vectors)


    dt = [('key', object), ('pri_deg', object), ('sec_deg', object), ('quality', object), ('inversion', object)]
    return np.array(one_hot_vectors, dtype=dt)

def translate_degree(degree_str):

    if ('+' not in degree_str and '-' not in degree_str) or ('+' in degree_str and degree_str[1] == '+'):
        degree_hot = int(degree_str[0])
    elif degree_str[0] == '-':
        degree_hot = int(degree_str[1]) + 14
    elif degree_str[0] == '+':
        degree_hot = int(degree_str[1]) + 7

    return degree_hot

def augment_tchords(labels_onehot):
    """
    Augment chord labels (in one-hot representation)
    :param labels_onehot:
    :return: labels_aug
    """

    labels_aug = [None for _ in range(12)]
    for m in range(len(labels_aug)):
        temp = np.array(labels_onehot)
        for i in range(temp.shape[0]):
            if temp[i][24] != 1:
                key = list(temp[i][:12]) if any(temp[i][:12]) else list(temp[i][12:24])
                mode = 0 if any(temp[i][:12]) else 1
                if m < 7:
                    shift = m
                else:
                    shift = m - 12
                temp[i] = list(np.roll(key, shift=shift)) + [0 for _ in range(12)] + [0] if mode == 0 \
                    else [0 for _ in range(12)] + list(np.roll(key, shift=shift)) + [0]
        labels_aug[m] = temp

    return labels_aug

def augment_rchords(labels_onehot):

    labels_aug = [None for _ in range(12)]
    for m in range(len(labels_aug)):
        temp = np.array(labels_onehot)
        for i in range(temp.shape[0]):
            key = list(temp[i]['key'][:12]) if any(temp[i]['key'][:12]) else list(temp[i]['key'][12:])
            mode = 0 if any(temp[i]['key'][:12]) else 1 # major -> 0, minor -> 1
            if m < 7:
                shift = m
            else:
                shift = m - 12
            temp[i]['key'] = list(np.roll(key, shift=shift)) + [0 for _ in range(12)] if mode == 0 else [0 for _ in range(12)] + list(np.roll(key, shift=shift))
        labels_aug[m] = temp

    return labels_aug

def prepare_input_data(segments_pianoroll, segments_label, hop=32, num_steps=64, feature_size=61*32, label_type=None):
    """
    Rearrange segments_pianoroll and segments_label into the format [num_sequences, num_steps, feature_size] and [num_sequences, num_steps, num_classes] respectively
    :param segments_pianoroll:
    :param segments_label:
    :param hop: hop size of sequences, default = 32 (4 beats)
    :param num_steps: number of RNN time steps
    :param feature_size: input feature size
    :param label_type: string, 'chord_symbol' and 'chord_function'  are valid
    :return: input_segments, input_labels
    """

    input_segments = [[None for _ in range(32)] for _ in range(12)]
    input_labels = [[None for _ in range(32)] for _ in range(12)]
    for m in range(12):
        for p in range(32):
            indices = list(range(len(segments_pianoroll[m][p]))) # indices of segments in the piece with m modulation
            seq_indices = [indices[i:i + num_steps] for i in itertools.takewhile(lambda x: x + num_steps < len(indices), range(0, len(indices), hop))] # split indices into sequences of length n_steps with hop size = hop
            if (len(indices) - num_steps) / hop != 0:
                seq_indices.append(indices[-num_steps:])
            num_sequences = len(seq_indices)

            if label_type == 'chord_symbol':
                inputs = np.zeros(shape=(num_sequences, num_steps, feature_size), dtype=np.float32)
                labels = np.zeros(shape=(num_sequences, num_steps), dtype=np.int32)
                for n in range(num_sequences):
                    inputs[n, :, :] = [segments_pianoroll[m][p][index] for index in seq_indices[n]]
                    labels[n, :] = [np.argmax(vector) for vector in segments_label[m][p][seq_indices[n]]]
            elif label_type == 'chord_function':
                inputs = np.zeros(shape=(num_sequences, num_steps, feature_size), dtype=np.float32)
                dt = [('key', 'int'), ('pri_deg', 'int'), ('sec_deg', 'int'), ('quality', 'int'), ('inversion', 'int')]
                labels = np.zeros(shape=(num_sequences, num_steps), dtype=dt)
                for n in range(num_sequences):
                    inputs[n, :, :] = [segments_pianoroll[m][p][index] for index in seq_indices[n]]
                    labels[n, :]['key'] = [np.argmax(vector) for vector in segments_label[m][p]['key'][seq_indices[n]]]
                    labels[n, :]['pri_deg'] = [np.argmax(vector) for vector in segments_label[m][p]['pri_deg'][seq_indices[n]]]
                    labels[n, :]['sec_deg'] = [np.argmax(vector) for vector in segments_label[m][p]['sec_deg'][seq_indices[n]]]
                    labels[n, :]['quality'] = [np.argmax(vector) for vector in segments_label[m][p]['quality'][seq_indices[n]]]
                    labels[n, :]['inversion'] = [np.argmax(vector) for vector in segments_label[m][p]['inversion'][seq_indices[n]]]
            else:
                print('LabelTypeError: %s,' % label_type, 'label_type should be \'chord_symbol\' or \'chord_function\'.')
                quit()
            input_segments[m][p] = inputs
            input_labels[m][p] = labels

    return input_segments, input_labels

def split_input_data(input_segments, input_labels):

    # split 32 pieces into three sets
    train_indices = [4, 11, 16, 20, 26, 31, 3, 8, 12, 17, 23, 21, 27, 29, 30, 10, 1, 2]
    valid_indices = [7, 18, 28, 15, 25, 5, 19]
    test_indices = [0, 13, 22, 14, 19, 24, 6]

    inputs_train = np.concatenate([input_segments[m][p] for m in range(12) for p in train_indices], axis=0)
    inputs_valid = np.concatenate([input_segments[0][p] for p in valid_indices], axis=0)
    inputs_test = np.concatenate([input_segments[0][p] for p in test_indices], axis=0)

    labels_train = np.concatenate([input_labels[m][p] for m in range(12) for p in train_indices], axis=0)
    labels_valid = np.concatenate([input_labels[0][p] for p in valid_indices], axis=0)
    labels_test = np.concatenate([input_labels[0][p] for p in test_indices], axis=0)

    return inputs_train, inputs_valid, inputs_test, labels_train, labels_valid, labels_test

def get_training_data(label_type=None):
    """
    x is input data, y is label;
    x has the shape [num_sequences, num_steps, feature_size];
    if label_type == 'chord_symbol',
        y has the shape [num_sequences, num_steps];
    if label_type == 'chord_function',
        y has the shape [num_sequences, num_steps],
        and chord functions can be access by y[num_sequences, num_steps][function_name],
        where 'key', 'pri_deg', 'sec_deg', 'quality', 'inversion' are valid function_name
    """

    print("Preprocessing the BPS-FH dataset:")

    if label_type not in ['chord_symbol', 'chord_function']:
        print('LabelTypeError: %s,' % label_type, 'label_type should be \'chord_symbol\' or \'chord_function\'.')
        quit()

    path = os.path.dirname(os.path.abspath(__file__)) + '\\BPS_FH_Dataset\\'
    print('load data...')
    pianorolls, tdeviation = load_notes(directory=path)
    chord_labels = load_chord_labels(directory=path)
    print('segment data...')
    segments_pianoroll, segments_label = segment_pianorolls(pianorolls, tdeviation, chord_labels, wsize=32, hsize=4, label_type=label_type)
    print('prepare data...')
    input_segments, input_labels = prepare_input_data(segments_pianoroll, segments_label, hop=32, num_steps=64, feature_size=1952, label_type=label_type)
    print('split data...')
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_input_data(input_segments, input_labels)

    lens = [len(segments) for segments in segments_pianoroll[0]]
    total_segments = sum([sum([len(segments) for segments in segments_pianoroll[m]]) for m in range(len(segments_pianoroll))])
    # total_labels = sum([sum([len(labels) for labels in segments_label[m]]) for m in range(len(segments_label))])
    print('num_segments in each piece =', lens)
    print('total_segments =', total_segments)
    sets = [x_train, x_valid, x_test, y_train, y_valid, y_test]
    print('shape_of_all_sets =', [set.shape for set in sets])

    return sets

if __name__ == '__main__':

    """
    x: the input data with shape = [num_sequences, num_steps, feature_size]
    y: the ground truth with shape = [num_sequences, num_steps]
    label_type: 'chord_symbol' for STL_BLSTM_RNNModel, and 'chord_function' for MTL_BLSTM_RNNModel 
    """
    [x_train, x_valid, x_test, y_train, y_valid, y_test] = get_training_data(label_type='chord_symbol')










