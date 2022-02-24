from Lexelt import get_data, get_key
f = open("C:/Users/rrros/OneDrive/Documents/COMPSCI/CMPU366/senseval3/train/EnglishLS.train", "r") 
train_data = get_data(f)
train_data.keys()
train_data["win.v"]

i = train_data["win.v"].get("win.v.bnc.00004762")
i.words
i.heads
i.words[56]

answer_f = open("C:/Users/rrros/OneDrive/Documents/COMPSCI/CMPU366/senseval3/train/EnglishLS.train.key", "r") 
get_key(answer_f, train_data)
i.answers