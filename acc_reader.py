import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fm = None
with open("merge_result.txt", "r") as f:
    fm = f.readlines()

fma = open("metric_acc.txt", "w")
fca = open("classifier_acc.txt", "w")
fga = open("grouping_acc.txt", "w")

ma_sum = 0
ca_sum = 0
ga_sum = 0
times = 0

for lines in fm:
    sp = lines.split()
    if sp[0] == "Metric":
        ma_sum += float(sp[2])
        print(lines, file=fma)
    elif sp[0] == "Classifier":
        ca_sum += float(sp[2])
        print(lines, file=fca)
    elif sp[0] == "Label":
        ga_sum += float(sp[4])
        times += 1
        print(lines, file=fga)

final_ma = ma_sum / (float)(times)
print(f"Average accuracy: {final_ma}", file=fma)
final_ca = ca_sum / (float)(times)
print(f"Average accuracy: {final_ca}", file=fca)
final_ga = ga_sum / (float)(times)
print(f"Average accuracy: {final_ga}", file=fga)


