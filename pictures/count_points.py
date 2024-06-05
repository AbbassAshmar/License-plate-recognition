result = {
    "CIN20356.jpg": "CIN20356",
    "CMG21FG.jpg": "CMG21FG",
    "FSD23429.jpg": "FSD23429",
    "PCT15PY.jpg": "PCT15PY",
    "PGN141GR.jpg": "PGN141GR",
    "PGN756EC.jpg": "SI756EC",
    "PKL8C63.jpg": "PKL8C63",
    "PKRR788.jpg": "PKRR788",
    "PKS30W3.jpg": "PKS30W3",
    "PO033AX.jpg": "PO033AX",
    "PO096NT.jpg": "PO096NT",
    "PO155KU.jpg": "PO155KU",
    "PO2J735.jpg": "PO2J735",
    "PO2W494.jpg": "PO2W494",
    "PO522WJ.jpg": "PO522WJ",
    "PO5T224.jpg": "PO5T224",
    "PO6K534.jpg": "PO6K534",
    "PO778SS.jpg": "PO778S5",
    "POBTC81.jpg": "POBTC81",
    "POZS221.jpg": "POZ5221",
    "PSE22800.jpg": "PSL22800",
    "PSZ47620.jpg": "PSZ47620",
    "PZ0460J.jpg": "PZ0460J",
    "PZ492AK.jpg": "PZ492AK",
    "WSCUP62.jpg": "WSCUP62",
    "ZSL17729.jpg": "ZSLI7729"
}

points = []
for key, value in result.items():
    cur_points = 0
    for i in range(len(value)):
        if value[i] == key[i]:
            cur_points += 1
    key_no_ext = key.split('.')[0]
    if key_no_ext == value:
        cur_points += 3
    points.append(cur_points)

print(points)
print(sum(points))