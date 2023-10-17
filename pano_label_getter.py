import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
from collections import Counter

def PanoLabelGetter(house_data, partition_data, train_type, need_to_cut = False):
    # out_path_stat = "pano_count" + train_type + ".txt"
    # fd = open(out_path_stat, 'w')
    label_pairs = []            # store img paths
    building_room_pairs = []
    original_pano_count_list = []
    grouping_list = []
    room_counter = {}
    # j = 0 
    first = False
    # print(train_type, file=fd)
    print(train_type)
    for id, houses in house_data.items():
        # print(id)
        if id in partition_data[train_type]:
            if first:
                print(id)
                # first = False
            i = 0
            room_pairs = []
            grouping_pairs = []
            for floor_id, floor_data in houses.items():
                room_label_a = floor_id.split("_")[1]
                for complete_room_data in floor_data.values():
                    for partial_room_id, partial_room_data in complete_room_data.items():
                        room_label_b = partial_room_id.split("_")[2]
                        for pano_id, pano_data in partial_room_data.items():
                            room_label = room_label_a + room_label_b
                            pano_path = os.path.join(id, pano_data["image_path"])
                            label = pano_data["new_label"]
                            primary_bool = pano_data["is_primary"]
                            label_pairs.append([pano_path, label])
                            room_pairs.append([pano_path, room_label])
                            grouping_pairs.append([pano_path, label, room_label, primary_bool, id])
                            i += 1

            # if filter_size == 0:
            #     continue
            # fix label here
            # transform room_label 4-code to 0~n   
            # print(grouping_list)
            j = 0
            new_labels = []
            transdict = {}
            for idx, room in enumerate(room_pairs):
                if room[1] not in transdict:
                    transdict[room[1]] = j
                    j += 1
                new_labels.append(transdict[room[1]])
                if first:
                    print(room[1], transdict[room[1]])
                room[1] = transdict[room[1]]
                # print(grouping_list[idx][2])
                grouping_pairs[idx][2] = transdict[grouping_pairs[idx][2]]
            
            # print(grouping_list)
            
            # print(f"ID: {id}, pano數: {i}, label數(有幾間partial room): {j}")
            grouping_list.append(grouping_pairs)

            # def PanoCutter():

            room_pairs_2 = []
            new_labels_2 = []

            def batchCutter(argu_room_pairs: list):
                min_label = min(argu_room_pairs, key = lambda x: x[1])[1]
                max_label = max(argu_room_pairs, key = lambda x: x[1])[1]
                batch_count = 6
                while max_label - min_label + 1 >= 2 * batch_count:
                    new_room_pair = [k for k in argu_room_pairs if k[1] >= min_label and k[1] < min_label + batch_count]
                    argu_room_pairs = [k for k in argu_room_pairs if k[1] >= min_label + batch_count]
                    min_label = min(argu_room_pairs, key = lambda x: x[1])[1]
                    building_room_pairs.append(new_room_pair)
                return argu_room_pairs

            if i > 72:
                # pano count > 72 => cut to half
                room_pairs_2 += room_pairs[i//2:]
                new_labels_2 += new_labels[i//2:]
                room_pairs = room_pairs[:i//2]
                new_labels = new_labels[:i//2]

                # Expand panos
                room_pairs_2, new_labels_2 = PanoExpander(room_pairs_2, new_labels_2)
                room_pairs_2 = sorted(room_pairs_2, key = lambda x: x[1])
                

                # need to let pos / neg pair balance
                # cut label count -> 1 ~ 11
                if need_to_cut:

                    room_pairs_2 = batchCutter(room_pairs_2)

                building_room_pairs.append(room_pairs_2)

            room_pairs, new_labels = PanoExpander(room_pairs, new_labels)
            room_pairs = sorted(room_pairs, key = lambda x: x[1])
            if need_to_cut:
                room_pairs = batchCutter(room_pairs)

            building_room_pairs.append(room_pairs)
            first = False

    # print(room_counter)
    return label_pairs, building_room_pairs, original_pano_count_list, grouping_list

def PanoExpander(pano_list: list, label_list: list):
    # get each label count
    count = dict(Counter(label_list))
    # expand pano which count < 2 (no positive pair)
    while(min(count.values()) <= 1):
        min_keys = [k for k, v, in count.items() if v == min(count.values())]
        roll_label = random.choice(min_keys)            # random get min label
        roll_index = label_list.index(roll_label)       # get label's index
        pano_list.append(pano_list[roll_index])         # duplicate pano
        label_list.append(roll_label)                   # duplicate label
        count = dict(Counter(label_list))
    return pano_list, label_list