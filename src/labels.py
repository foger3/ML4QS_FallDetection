# labeling
label_map = dict()
label_categories = [
    "Label Walking",
    "Label Running",
    "Label Kneeling",
    "Label Lying",
    "Label Sitting",
    "Label Standing",
    "Label Falling",
    "Error",
]

# round1-luca
labels = [
    label_categories[0],
    label_categories[1],
    label_categories[4],
    label_categories[5],
    label_categories[6],
    label_categories[3],
    label_categories[2],
    label_categories[2],
    label_categories[4],
    label_categories[1],
    label_categories[0],
    label_categories[6],
    label_categories[3],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[7],
    label_categories[2],
    label_categories[6],
    label_categories[7],
]
label_map["round1-luca"] = labels
# round1-nicole
labels = [
    label_categories[0],
    label_categories[1],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[4],
    label_categories[6],
    label_categories[7],
    label_categories[7],
    label_categories[5],
    label_categories[4],
    label_categories[6]
]
label_map["round1-nicole"] = labels
# round1-sam
labels = [
    label_categories[0],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[1],
    label_categories[2],
    label_categories[3],
    label_categories[6],
    label_categories[5],
    label_categories[6],
    label_categories[2],
    label_categories[0],
    label_categories[6],
    label_categories[4]
]
label_map["round1-sam"] = labels
# round2-luca
labels = [
    label_categories[5],
    label_categories[0],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[3],
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[5],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[2]
]
label_map["round2-luca"] = labels
# round2-nicole
labels = [
    label_categories[4],
    label_categories[2],
    label_categories[3],
    label_categories[0],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[0],
    label_categories[7],
    label_categories[3],
]
label_map["round2-nicole"] = labels
# round3-nicole
labels = [
    label_categories[0],
    label_categories[4],
    label_categories[6],
    label_categories[6],
    label_categories[3],
    label_categories[6],
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[1],
    label_categories[4],
    label_categories[7],
    label_categories[6],
    label_categories[0],
    label_categories[6],
    label_categories[6],
    label_categories[5],
    label_categories[3],
    label_categories[6],
    label_categories[2],
    label_categories[6],
]
label_map["round3-nicole"] = labels
# round3-sam
labels = [
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[0],
    label_categories[6],
    label_categories[7],
    label_categories[3],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[5],
    label_categories[5],
    label_categories[3],
    label_categories[6],
    label_categories[0],
]
label_map["round3-sam"] = labels