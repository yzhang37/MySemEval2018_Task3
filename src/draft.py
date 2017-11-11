import re
import socket
from src.stanfordCoreNLP import StanfordCoreNLP
def regex():
    str = "I have three test and a two dance performances tomorrow!!:books::open_book::dancer::dancer::woman_with_bunny_ears:  #EasyDay "
    emoji_re = r':[a-z|-|_]+:'
    emoji_list = []

    a = re.findall(emoji_re,str)
    # b = re.indexof(emoji_re,str)
    for i in re.finditer(emoji_re, str):
        emoji_list.append((i.group(),i.span()))
        print(i.group(),i.span())
        # print(type(i.span()))

    # print(emoji_list)

    text = re.sub(emoji_re, "", str)
    print(str)
    print(text)

print(socket.gethostname())
