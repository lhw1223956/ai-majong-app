TILE_INFO = {
    '1w': {'name': '一萬', 'icon': '🀇', 'w': 1, 'type': 'w', 'val': 1}, '2w': {'name': '二萬', 'icon': '🀈', 'w': 2, 'type': 'w', 'val': 2},
    '3w': {'name': '三萬', 'icon': '🀉', 'w': 3, 'type': 'w', 'val': 3}, '4w': {'name': '四萬', 'icon': '🀊', 'w': 4, 'type': 'w', 'val': 4},
    '5w': {'name': '五萬', 'icon': '🀋', 'w': 5, 'type': 'w', 'val': 5}, '6w': {'name': '六萬', 'icon': '🀌', 'w': 6, 'type': 'w', 'val': 6},
    '7w': {'name': '七萬', 'icon': '🀍', 'w': 7, 'type': 'w', 'val': 7}, '8w': {'name': '八萬', 'icon': '🀎', 'w': 8, 'type': 'w', 'val': 8},
    '9w': {'name': '九萬', 'icon': '🀏', 'w': 9, 'type': 'w', 'val': 9},
    '1D': {'name': '一筒', 'icon': '🀙', 'w': 11, 'type': 'D', 'val': 1}, '2D': {'name': '二筒', 'icon': '🀚', 'w': 12, 'type': 'D', 'val': 2},
    '3D': {'name': '三筒', 'icon': '🀛', 'w': 13, 'type': 'D', 'val': 3}, '4D': {'name': '四筒', 'icon': '🀜', 'w': 14, 'type': 'D', 'val': 4},
    '5D': {'name': '五筒', 'icon': '🀝', 'w': 15, 'type': 'D', 'val': 5}, '6D': {'name': '六筒', 'icon': '🀞', 'w': 16, 'type': 'D', 'val': 6},
    '7D': {'name': '七筒', 'icon': '🀟', 'w': 17, 'type': 'D', 'val': 7}, '8D': {'name': '八筒', 'icon': '🀠', 'w': 18, 'type': 'D', 'val': 8},
    '9D': {'name': '九筒', 'icon': '🀡', 'w': 19, 'type': 'D', 'val': 9},
    '1s': {'name': '一條', 'icon': '🀐', 'w': 21, 'type': 's', 'val': 1}, '2s': {'name': '二條', 'icon': '🀑', 'w': 22, 'type': 's', 'val': 2},
    '3s': {'name': '三條', 'icon': '🀒', 'w': 23, 'type': 's', 'val': 3}, '4s': {'name': '四條', 'icon': '🀓', 'w': 24, 'type': 's', 'val': 4},
    '5s': {'name': '五條', 'icon': '🀔', 'w': 25, 'type': 's', 'val': 5}, '6s': {'name': '六條', 'icon': '🀕', 'w': 26, 'type': 's', 'val': 6},
    '7s': {'name': '七條', 'icon': '🀖', 'w': 27, 'type': 's', 'val': 7}, '8s': {'name': '八條', 'icon': '🀗', 'w': 28, 'type': 's', 'val': 8},
    '9s': {'name': '九條', 'icon': '🀘', 'w': 29, 'type': 's', 'val': 9},
    'ew': {'name': '東', 'icon': '🀀', 'w': 31, 'type': 'z'}, 'sw': {'name': '南', 'icon': '🀁', 'w': 32, 'type': 'z'},
    'ww': {'name': '西', 'icon': '🀂', 'w': 33, 'type': 'z'}, 'nw': {'name': '北', 'icon': '🀃', 'w': 34, 'type': 'z'},
    'zhong': {'name': '中', 'icon': '🀄︎', 'w': 35, 'type': 'z'}, 'fa': {'name': '發', 'icon': '🀅', 'w': 36, 'type': 'z'},
    'wd': {'name': '白', 'icon': '🀆', 'w': 37, 'type': 'z'},
    '1rf': {'name': '春', 'icon': '🀦', 'w': 51, 'type': 'h', 'suit': 'rf', 'v': 1}, '2rf': {'name': '夏', 'icon': '🀧', 'w': 52, 'type': 'h', 'suit': 'rf', 'v': 2},
    '3rf': {'name': '秋', 'icon': '🀨', 'w': 53, 'type': 'h', 'suit': 'rf', 'v': 3}, '4rf': {'name': '冬', 'icon': '🀩', 'w': 54, 'type': 'h', 'suit': 'rf', 'v': 4},
    '1bf': {'name': '梅', 'icon': '🀢', 'w': 55, 'type': 'h', 'suit': 'bf', 'v': 1}, '2bf': {'name': '蘭', 'icon': '🀣', 'w': 56, 'type': 'h', 'suit': 'bf', 'v': 2},
    '3bf': {'name': '竹', 'icon': '🀤', 'w': 57, 'type': 'h', 'suit': 'bf', 'v': 3}, '4bf': {'name': '菊', 'icon': '🀥', 'w': 58, 'type': 'h', 'suit': 'bf', 'v': 4}
}

CODE_TO_IDX = {
    '1w':0, '2w':1, '3w':2, '4w':3, '5w':4, '6w':5, '7w':6, '8w':7, '9w':8,
    '1D':9, '2D':10, '3D':11, '4D':12, '5D':13, '6D':14, '7D':15, '8D':16, '9D':17,
    '1s':18, '2s':19, '3s':20, '4s':21, '5s':22, '6s':23, '7s':24, '8s':25, '9s':26,
    'ew':27, 'sw':28, 'ww':29, 'nw':30, 'zhong':31, 'fa':32, 'wd':33
}

IDX_TO_CODE = {v: k for k, v in CODE_TO_IDX.items()}
