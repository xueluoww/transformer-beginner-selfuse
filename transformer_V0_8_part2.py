import transformer_V0_8_part1 as core  
import torch
from torch.utils.data import DataLoader
from pathlib import Path

#建立总词表  
currendir = Path(__file__).resolve().parent
vocalpathc = currendir / "vocalchinese"
vocalpathe = currendir / "vocalenlish"
modelpath = currendir / "new_model.pth"
textpath = currendir / "text.txt"
datasetr = core.data(textpath)
dataloaderr = DataLoader(
    datasetr,
    batch_size = 1,
    shuffle= False,
    num_workers= 0
)
textc = "" 
texte = ""
for dataa in dataloaderr:
    textc = textc +"".join(dataa["encodertext"])
    texte = texte +"".join(dataa["decodertext"])
if vocalpathc.exists() and vocalpathe.exists():
    print("开始读取词表喵")
    id2tokenc,dictc= core.loadjson(vocalpathc) 
    id2tokene,dicte= core.loadjson(vocalpathe) 
elif not vocalpathc.exists() and not vocalpathe.exists():
    print("开始建立词表喵")
    dictc,dicte,id2tokenc,id2tokene= core.bulievocab(textc,texte,vocalpathc,vocalpathe) 
else:
    raise FileNotFoundError(
        "不行训不了，词库出问题了"
    )
lenid2tokenc = len(id2tokenc)
lenid2tokene = len(id2tokene)
model = core.reallytrainmodel(lenid2tokenc = lenid2tokenc,lenid2tokene = lenid2tokene)
if modelpath.exists() :
    model.load_state_dict(torch.load(modelpath,weights_only = True))
else:
    print("没找到参数喵，好像还没训练过喵，开始第一次训练喵")

modechoose = input("输入train进入30轮训练,输入translate进入翻译模式").strip()

if modechoose =="train":
#训练部分
    model.train()
    trainloader = DataLoader(
        datasetr,
        batch_size = 8,
        shuffle = True,
        num_workers= 0
    )
    lossfun = torch.nn.CrossEntropyLoss(ignore_index = 0)
    op = torch.optim.AdamW(model.parameters(),lr = 1e-4,weight_decay = 0 )
    for i in range(30):
        losssum = 0.0
        tokennumber = 0
        print("第{}轮训练".format(i+1))
        for dataa in trainloader:
            traintextc = dataa["encodertext"]
            traintexte = dataa["decodertext"]
            tokenencoder,tokendecoder,tokenanswer= core.myinput4train(traintextc,traintexte,dictc = dictc,dicte = dicte)
            op.zero_grad()
            logits = model(tokenencoder,tokendecoder)
            logits = logits.reshape(-1,logits.shape[-1])
            tokenanswer = tokenanswer.reshape(-1)
            vildtokennumber = (tokenanswer != core.padid).sum().item()
            loss = lossfun(logits,tokenanswer)
            losssum = losssum + loss.item() * vildtokennumber
            tokennumber = tokennumber + vildtokennumber
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = 1.0)
            op.step()
        torch.save(model.state_dict(),modelpath)
        average = losssum / tokennumber
        allloss = losssum
        print("本轮总loss:{:.4f}".format(allloss))
        print("本轮平均loss:{:.4f}".format(average))
        print("训练完毕")

elif modechoose == "translate":
#推理部分
    model.eval()
    textc = input("请输入要翻译的文本").strip()
    encodertoken,decodertoken = core.myinput4inference(textc,dictc,dicte)
    core.inference(core.maxcoutlen,model,encodertoken,decodertoken,id2tokene)