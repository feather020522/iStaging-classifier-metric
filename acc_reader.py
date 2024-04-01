import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fma = None
with open("./merge_result_single/metric_acc.txt", "r") as f:
    fma = f.readlines()

fmga = None
with open("./merge_result_single/metric_group_acc.txt", "r") as f2:
    fmga = f2.readlines()

# fma = open("metric_acc.txt", "w")
# # fmga = open("metric_group_acc.txt", "w")
# fca = open("classifier_acc.txt", "w")
# fga = open("grouping_acc.txt", "w")

ma_sum = 0
mga_sum = 0
ca_sum = 0
ga_sum = 0
times = 0

for lines in fma:
    sp = lines.split()
    if sp[0] == "Metric":
        ma_sum += float(sp[2])
        # print(lines, file=fma)
        times += 1

for lines in fmga:
    sp = lines.split()
    if sp[0] == "Grouping":
        mga_sum += float(sp[2])
        # print(lines, file=fma)
        # times += 1
    # elif sp[0] == "Classifier":
        # ca_sum += float(sp[2])
        # print(lines, file=fca)
    # elif sp[0] == "Label":
        # ga_sum += float(sp[4])
        # print(lines, file=fga)

final_ma = ma_sum / (float)(times)
print(f"Average accuracy: {final_ma}")
final_mga = mga_sum / (float)(times)
print(f"Average accuracy: {final_mga}")
# final_ca = ca_sum / (float)(times)
# print(f"Average accuracy: {final_ca}", file=fca)
# final_ga = ga_sum / (float)(times)
# print(f"Average accuracy: {final_ga}", file=fga)


