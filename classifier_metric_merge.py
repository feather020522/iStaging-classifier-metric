# thoughts
# 取panorama, room_id, label
# 第一段先分好組, getitem拿pano + room_id
# 第二段分label, getitem拿pano + label
# 依第一段結果分組後, 看誰是primary panorama
# 判斷其他secondary panorama的label是不是大都跟primary一樣
# 完成分組