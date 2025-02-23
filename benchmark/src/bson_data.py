import bson

def load_bson_file(file_path):
    with open(file_path, "rb") as f:
        data = bson.decode_all(f.read())  
    return data

pins = load_bson_file("../pinterest_iccv/subset_iccv_pin_im.bson")  # Pins and images
boards = load_bson_file("../pinterest_iccv/subset_iccv_board_pins.bson")  # Boards & pins
categories = load_bson_file("../pinterest_iccv/subset_iccv_board_cate.bson")  # Board categories


# Pins:
# _id
# im_url
# im_name
# pin_id

# Boards:
# _id
# board_url
# board_id
# pins
#   [List of <class 'str'>]

# Categories:
# _id
# board_id
# board_url
# cate_id