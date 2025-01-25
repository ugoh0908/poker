class change_card_name:
    def change_card_name(self, origin_name):
        split_name = origin_name.split("_")
        if split_name[0] == "1":
            split_name[0] = "A"
        elif split_name[0] == "10":
            split_name[0] = "T"
        if split_name[1] == "diamond":
            split_name[1] = "d"
        elif split_name[1] == "heart":
            split_name[1] = "h"
        elif split_name[1] == "clover":
            split_name[1] = "c"
        else:
            split_name[1] = "s"
        return split_name[0] + split_name[1]
