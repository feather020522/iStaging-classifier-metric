import argparse
import json
import os

# import tqdm as tqdm
import numpy as np

def make_output_json(input_folder, tour_ids_split):
    output_dic = {}
    label_dic = {}
    organized_label_dic = {}
    
    for tour_id in tour_ids_split:
        tour_json_path = os.path.join(input_folder, tour_id, "zind_data.json")
        with open(tour_json_path, "r") as fh:
            zillow_json_dict = json.load(fh)

        house_dic = {}
        num_floors = 0
        num_primary_panos = 0
        num_secondary_panos = 0
        # num_complete_rooms = 0
        # num_partial_rooms = 0

        if "merger" in zillow_json_dict:
            for floor_id, floor_data in zillow_json_dict["merger"].items():
                num_floors += 1
                floor_dic = {}
                # floor_dic["num_complete_rooms"] = len(floor_data)

                for complete_num, complete_room_data in floor_data.items():
                    # num_complete_rooms += 1
                    complete_room_dic = {} 
                    # complete_room_dic["num_partial_rooms"] = len(complete_room_data) 

                    for partial_num, partial_room_data in complete_room_data.items():
                        # num_partial_rooms += 1
                        partial_room_dic = {}
                        partial_room_primary_num = 0
                        partial_room_secondary_num = 0
                        # partial_room_dic["num_panos"] = len(partial_room_data)

                        for pano_num, pano_data in partial_room_data.items():
                            

                            label = pano_data["label"]
                            new_label = label
                            # new_label = label_resign(label)
                            # print(label, new_label)

                            pano_dic = {}
                            pano_dic["is_primary"] = pano_data["is_primary"]
                            pano_dic["is_inside"] = pano_data["is_inside"]
                            pano_dic["label"] = label
                            pano_dic["new_label"] = new_label
                            pano_dic["image_path"] = pano_data["image_path"]
                            pano_dic["floor_number"] = pano_data["floor_number"]
                            partial_room_dic[pano_num] = pano_dic
                            if pano_data["is_primary"]:
                                if label_dic.get(label) == None:
                                    label_dic[label] = 1
                                    # print (label, tour_id)
                                else:
                                    label_dic[label] += 1
                                
                                if organized_label_dic.get(new_label) == None:
                                    organized_label_dic[new_label] = 1
                                else:
                                    organized_label_dic[new_label] += 1

                                num_primary_panos += 1
                                # partial_room_primary_num += 1
                            else:
                                num_secondary_panos += 1
                                # partial_room_secondary_num += 1

                            # partial_room_dic["num_primary_panos"] = partial_room_primary_num
                            # partial_room_dic["num_secondary_panos"] = partial_room_secondary_num
                        complete_room_dic[partial_num] = partial_room_dic   
                    floor_dic[complete_num] = complete_room_dic
                house_dic[floor_id] = floor_dic

        
        # house_dic["num_floors"] = num_floors

        # need these?
        # house_dic["num_primary_panos"] = num_primary_panos
        # house_dic["num_secondary_panos"] = num_secondary_panos
        
        output_dic[tour_id] = house_dic

    return output_dic, label_dic, organized_label_dic

def label_resign(label):
    if label.find("closet") != -1:              # done
        return "closet"
    elif label.find("entrance") != -1:          # like doorway/stairs, ex 0891, pano 05 & 60
        return "doorway"
    elif label.find("hallway") != -1:           # done
        return "hallway"
    elif label.find("hall") != -1:              # done
        return "hallway"
    elif label.find("landing") != -1:           # done
        return "stair"
    elif label.find("entry") != -1:             # like doorway, ex 0336, pano 54
        return "doorway"
    elif label.find("stair") != -1:             # done
        return "stair"
    
    if label.find("patio") != -1:               # done
        return "patio"
    
    if label.find("kitchen") != -1:             # done
        return "kitchen"
    
    if label.find("basement") != -1:            # done
        return "basement"
    
    if label.find("pantry") != -1:              # done
        return "pantry"
    
    if label.find("attic") != -1:               # done
        return "attic"
    
    if label.find("laundry") != -1:             # done
        return "laundry"
    
    if label.find("yard") != -1:                # done
        return "yard"
    
    if label.find("boiler room") != -1:         # too few, merge into other, 0973, pano 15
        return "other"

    if label.find("dining room") != -1:
        return "dining room"
    
    if label.find("shower") != -1:              # done
        return "bathroom"
    
    if label.find("utilities") != -1:           # done
        return "utility"
    
    if label.find("storage") != -1:             # == utility
        return "utility"
    
    if label.find("primary bathroom") != -1:    # don't distribute primary bathroom & bathroom
        return "bathroom"
    elif label.find("bathroom") != -1:
        return "bathroom"
    elif label.find("bath") != -1:
        return "bathroom"
    
    if label.find("primary bedroom") != -1:
        return "bedroom"
    elif label.find("bedroom") != -1:
        return "bedroom"
    elif label.find("bed") != -1:
        return "bedroom"

    if label.find("hide") != -1:                # like hallway, 0227, pano 48
        return "hallway"
    if label.find("hidden scan") != -1:         # like hallway, 0590, pano 16
        return "hallway"
    
    if label.find("hidden") != -1:              # like doorway, 0368, pano 65
        return "doorway"
    
    if label.find("doorway") != -1:             
        return "doorway"
    elif label.find("door") != -1:              # example 0006, pano 48
        return "doorway"
    
    if label.find("frame") != -1:               # like doorway, 1282, pano 47
        return "doorway"
    
    if label.find("fireplace") != -1:           # like doorway, 0110, pano 4
        return "doorway"
    
    if label.find("lobby") != -1:               # looks like doorway, 0373
        return "doorway"
    
    if label.find("foyer") != -1:               # looks like doorway, 0047, pano 3
        return "doorway"
    
    if label.find("car port") != -1:            # outside garage?, 1403
        return "garage"
    
    if label.find("porch") != -1:               # like hallway, connect doors, 0264, pano 29
        return "hallway"
    
    if label.find("floor map") != -1:           # like opening, 1197, pano 0
        return "hallway"
    
    if label.find("loser") != -1:               # like opening, 0340, pano 60 (why loser??)
        return "hallway"
    
    if label.find("floor plan calibration") != -1:      # like hallway, 0072, pano 0
        return "hallway"
    if label.find("floor plan") != -1:          # like hallway or doorway?, 0302, pano 0
        return "hallway"
    
    if label.find("playroom") != -1:            # An empty space, upstairs, 1195
        return "other"

    if label.find("space") != -1:
        return "other"
    
    if label.find("deck") != -1:                # like yard, but with roof
        return "other"
    
    if label.find("points") != -1:
        return "other"
    
    if label.find("bonus") != -1:               # simple room, 0000, pano 15
        return "other"
    
    if label.find("water") != -1:               # water heater
        return "other"

    if label.find("alley") != -1:
        return "hallway"

    if len(label) <= 2:
        return "other"
    
    return label

def collect_stats(input_folder, tour_ids_split):
    house_id = ""
    num_floors_list = []
    num_primary_pano_list = []
    num_secondary_pano_list = []
    num_inside_room_list = []
    house_room_list = []

    room_types = {}
    num_bedrooms = 0
    type_bedrooms_count = {}
    type_bedrooms_count["bedroom"] = 0

    for tour_id in tour_ids_split:
        tour_json_path = os.path.join(input_folder, tour_id, "zind_data.json")
        with open(tour_json_path, "r") as fh:
            zillow_json_dict = json.load(fh)
        
        house_id = tour_id
        num_floors = 0
        num_complete_rooms = 0
        num_partial_rooms = 0
        num_primary_panos = 0
        num_secondary_panos = 0
        num_inside_room = 0

        room_list = {}
        room_list["bedroom"] = 0
        room_list["contain bedroom"] = 0
        
        if "merger" in zillow_json_dict:
            for floor_id, floor_data in zillow_json_dict["merger"].items():
                num_floors += 1
                for complete_room_data in floor_data.values():
                    num_complete_rooms += 1
                    printed = 0
                    for partial_room_data in complete_room_data.values():
                        num_partial_rooms += 1
                        for pano_data in partial_room_data.values():
                            room_type = pano_data["label"]
                            if (room_types.get(room_type) == None):
                                # print(room_type, tour_id)
                                room_types[room_type] = 1
                            else:
                                room_types[room_type] += 1
                            # if pano_data["is_primary"]:
                            #     num_primary_panos += 1
                            #     if not printed:
                            #         room_type = pano_data["label"]
                            #         # collect bedrooms
                            #         if room_type == "bedroom":                              # "bedroom"
                            #             num_bedrooms += 1
                            #             room_list[room_type] += 1
                            #             type_bedrooms_count[room_type] += 1
                            #         elif room_type.find("bedroom") != -1:                   # contain "bedroom"
                            #             num_bedrooms += 1
                            #             if (type_bedrooms_count.get(room_type) == None):
                            #                 type_bedrooms_count[room_type] = 1
                            #             else:
                            #                 type_bedrooms_count[room_type] += 1
                            #             room_list["contain bedroom"] += 1
                            #         else:                                                   # others
                            #             if (room_list.get(room_type) == None):
                            #                 room_list[room_type] = 1
                            #             else:
                            #                 room_list[room_type] += 1
                                    
                            #         if (room_types.get(room_type) == None):
                            #             # print(room_type, tour_id)
                            #             room_types[room_type] = 1
                            #         else:
                            #             room_types[room_type] += 1
                            #         # else:
                            #         #     print(room_type, type(room_type))
                            #         printed = 1
                            # else:
                            #     num_secondary_panos += 1

                            # if pano_data["is_inside"]:
                            #     num_inside_room += 1

        
        num_floors_list.append(num_floors)
        num_primary_pano_list.append(num_primary_panos)
        num_secondary_pano_list.append(num_secondary_panos)
        house_room_list.append(room_list)
        num_inside_room_list.append(num_inside_room)
        # print(type_bedrooms_count)
    
    return [house_id, num_floors_list, num_primary_pano_list, num_secondary_pano_list, house_room_list, type_bedrooms_count, room_types, num_bedrooms, num_inside_room_list]

parser = argparse.ArgumentParser(description="Partition Zillow Indoor Dataset (ZInD)")

parser.add_argument(
    "--input", "-i", help="Input folder contains all the home tours.", required=True
)
parser.add_argument(
    "--output", "-o", help="Output folder where zind_partition.json will be saved to", required=True
)

args = parser.parse_args()
input_folder = args.input
output_folder = args.output
tour_ids = [tour_id for tour_id in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, tour_id))]

output_dic, label_dic, organized_label_dic = make_output_json(input_folder, tour_ids)

# print("\nlabels:")
# for types, count in label_dic.items():
#     print(f'    {types}: {count}')
# print(f'\nTotal: {len(label_dic)}')

print("\nlabels in primary panorama after organize:")
for types, count in organized_label_dic.items():
    print(f'    {types}: {count}')
print(f'\nTotal: {len(organized_label_dic)}')

# with open(os.path.join(output_folder, "stats.json"), "w") as outfile:
#     json.dump(output_dic, outfile)

# print("\nfile outputed to stats.json")

with open(os.path.join(output_folder, "labels.json"), "w") as outfile:
    json.dump(organized_label_dic, outfile)

print("\nlabel outputed to labels.json")

exit()

output_list = collect_stats(input_folder, tour_ids)

room_counter = 55

output_dic = {}
bedroom_counter_list = {0: 0, 1: 0, 2: 0, 3: 0, ">=4": 0}
complete_room_counter_list = [0] * room_counter
house_id = output_list[0]
num_floors_list = output_list[1]
num_primary_pano_list = output_list[2]
num_secondary_pano_list = output_list[3]
house_room_list = output_list[4]
type_bedrooms_count = output_list[5]
room_types = output_list[6]
num_bedrooms = output_list[7]
num_inside_room_list = output_list[8]

i = 0

# for room_list in house_room_list:           # type(room_list): list of dictionary
#     if (room_list.get("bedroom") != None):
#         # print(f'bedroom: {room_list["bedroom"]}')
#         if room_list["bedroom"] >= 4:
#             bedroom_counter_list[">=4"] += 1
#         else:
#             bedroom_counter_list[room_list["bedroom"]] += 1
#     else:
#         bedroom_counter_list[0] += 1

#     if (num_primary_pano_list[i] >= room_counter - 1):
#         complete_room_counter_list[room_counter - 1] += 1
#     else:
#         complete_room_counter_list[num_primary_pano_list[i]] += 1
    
#     temp_dic = {}

#     print(f'house {i:04}:')
#     temp_dic["num_floors"] = num_floors_list[i]
#     print(f'    num_floors: {num_floors_list[i]}')
#     temp_dic["num_primary_panos"] = num_primary_pano_list[i]
#     print(f'    num_primary_panos: {num_primary_pano_list[i]}')
#     temp_dic["num_secondary_panos"] = num_secondary_pano_list[i]
#     print(f'    num_secondary_panos: {num_secondary_pano_list[i]}')
#     temp_dic['num_inside_room'] = num_inside_room_list[i]
#     print(f'    num_inside_room: {num_inside_room_list[i]}')
#     temp_dic["room_list"] = room_list
#     print(f'    room list:')
#     for type, count in room_list.items():
#         print(f'        {type}: {count}')
#     output_dic[str("house " + str(i))] = temp_dic
#     i += 1

print("\nbedroom count:")
for n, c in bedroom_counter_list.items():
    print(f'{n}: {c}')
# print(bedroom_counter_list)

print("\ncomplete room count: ")
for i in range(room_counter):
    if i == room_counter - 1:
        print(f'>={room_counter - 1}: {complete_room_counter_list[i]}')
    else:
        print(f'{i}: {complete_room_counter_list[i]}')

print(f"\nroom types: {len(room_types)}")
i = 0
for types, count in room_types.items():
    # if count <= 100:
    if types.find("hallway") == -1 and types.find("closet") == -1 and types.find("landing") == -1 and types.find("hall") == -1 and types.find("patio") == -1 and types.find("entry") == -1:
        if types.find("stair") == -1 and types.find("kitchen") == -1 and types.find("basement") == -1 and types.find("entrance") == -1 and types.find("shower") == -1 and types.find("pantry") == -1:
            if types.find("laundry") == -1 and types.find("attic") == -1  and types.find("yard") == -1 and types.find("boiler room") == -1 and types.find("door") == -1:
                if types.find("bath") == -1 and types.find("bed") == -1 and types.find("hidden") == -1 and types.find("hide") == -1 and types.find("foyer") == -1:
                    if types.find("dining room") == -1 and types.find("space") == -1:    
                        if len(types) > 2:
                            print(f'    {types}: {count}')
                            i += 1

print(f'unfiltered: {i}')


print("")
print(f'total panorama contain bedroom: {num_bedrooms}')
print("bedroom list:")
for types, count in type_bedrooms_count.items():
    print(f'    {types}: {count}')
# print(type_bedrooms_count)
# print(room_types)
