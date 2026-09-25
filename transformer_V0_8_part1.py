from math import sin,cos
import torch
from torch import nn, softmax 
from torch.utils.data import Dataset
import json

#still huge update

#词表部分
padid = 0
bosid = 1
eosid = 2
unkid = 3

maxcoutlen = 100

class data(Dataset):
    def __init__(self,filepath):
        self.lists = []
        with open(filepath,mode="r",encoding="utf-8") as file:
            for line,text in enumerate(file,start = 1):
                text = text.rstrip("\n")
                if text == "":
                    continue
                parts = text.split("\t",1)
                if len(parts) != 2:
                    raise ValueError(
                        f"第{line}行没有使用tab分割,你做的什么破数据集"
                    )
                chinesetext = parts[0]
                enlishtext = parts[1]
                self.lists.append({
                    "encodertext" : chinesetext,
                    "decodertext" : enlishtext
                })

    def __len__(self):#句子数量
        return len(self.lists)

    def __getitem__(self,index):
        return self.lists[index]

def loadjson(filepth):
    with open(filepth,mode = 'r',encoding = "utf-8") as file:
        id2token = json.load(file)
        dict1 = dict()
        for i in range(len(id2token)):
            dict1[id2token[i]] = i
    return id2token,dict1

def xray(text,name):
    id2token = ['<PAD>','<BOS>','<EOS>','<UNK>']
    set1 = set(id2token)
    for i in range(len(text)):
        if text[i] not in set1:
            set1.add(text[i])
            id2token.append(text[i])
    with open(name,mode = 'w',encoding = "utf-8") as file :
        json.dump(id2token,file,ensure_ascii = False,indent = 2)
    dict1 = dict()
    for i in range(len(id2token)):
        dict1[id2token[i]] = i
    return  id2token,dict1   #扫描输入


def trans2token(textc,texte,dictc,dicte):#dict1 like token2id
    listencoder = list()
    maxencoderlength = 0
    for i in range(len(textc)) :
        temphave = []
        if len(textc[i])>maxencoderlength:
            maxencoderlength = len(textc[i])
        for j in range(len(textc[i])):
            if textc[i][j] not in dictc:
                temphave.append(3)
            else :
                temphave.append(dictc[textc[i][j]])
        listencoder.append(temphave)
    for i in range(len(textc)):
        for j in range(maxencoderlength - len(textc[i])):
            listencoder[i].append(0)
    tokenencoder = torch.tensor(listencoder,dtype = torch.long)
    tokenencoder = torch.unsqueeze(tokenencoder,1)

    listdecoder = list()
    listanswer = list()
    maxdecoderlength = 0
    for i in range(len(texte)) :
        temphave = []
        temphave.append(1)
        temphaveans = []
        if len(texte[i])>maxdecoderlength:
            maxdecoderlength = len(texte[i])
        for j in range(len(texte[i])):
            if texte[i][j] not in dicte:
                temphave.append(3)
                temphaveans.append(3)
            else :
                temphave.append(dicte[texte[i][j]])
                temphaveans.append(dicte[texte[i][j]])
        listdecoder.append(temphave)
        temphaveans.append(2)
        listanswer.append(temphaveans)
    for i in range(len(texte)):
        for j in range(maxdecoderlength - len(texte[i])):
            listdecoder[i].append(0)
            listanswer[i].append(0)
    tokendecoder = torch.tensor(listdecoder,dtype = torch.long)
    tokendecoder = torch.unsqueeze(tokendecoder,1)
    tokenanswer = torch.tensor(listanswer,dtype = torch.long)
    tokenanswer = torch.unsqueeze(tokenanswer,1)
    return tokenencoder,tokendecoder,tokenanswer    #返回token


#transformer逻辑部分
class trans2attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lineark=nn.Linear(512,512,bias= False)
        self.linearq=nn.Linear(512,512,bias= False)
        self.linearv=nn.Linear(512,512,bias= False)

    def trans(self,input):
        self.k=self.lineark(input)
        self.q=self.linearq(input)
        self.v=self.linearv(input)
        return self.k,self.v,self.q
    
    def forward(self,input):
        output =self.trans(input)
        return output

class decoderkvqget(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = nn.Linear(512,512)
        self.linear3 = nn.Linear(512,512)
        self.linear4 = nn.Linear(512,512)

    def forward(self,encdoer,maskattention):
        q = self.linear2(maskattention)
        k = self.linear3(encdoer)
        v = self.linear4(encdoer)
        return k,v,q     

class attention(nn.Module):
    def __init__(self):
        super().__init__()

    def jisuan(self,k,v,q,padding_mask = None):
        k1=k
        v1=v
        q1=q
        temp1 = k1.shape[-1] 
        k1 = k1.transpose(2,3)
        temp1 = pow(temp1,0.5)
        temp2 = (q1@k1)/temp1
        if padding_mask is not None:
            temp2 = temp2.masked_fill(
                padding_mask,
                float("-inf"),
            )
        out = softmax(temp2,dim=-1)@v1
        return out

    def forward(self,k,v,q,padding_mask = None):
        output = self.jisuan(k,v,q,padding_mask)
        return output  #单头attention只负责计算 


class maskattention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,k,v,q,padding_mask):
        k1=k
        v1=v
        q1=q
        temp1 = k1.shape[-1] 
        k1 = k1.transpose(2,3)
        temp1=pow(temp1,0.5)
        temp2 =(q1@k1)/temp1
        decoderlen = q.shape[2]
        helpmatrix = torch.triu(torch.full((decoderlen,decoderlen),float('-inf')),diagonal = 1)#创建三角矩阵
        temp2 = temp2+helpmatrix        
        if padding_mask is not None:
            temp2 = temp2.masked_fill(
                padding_mask,
                float("-inf"),
            )
        out=softmax(temp2,dim=-1)@v1
        return out

                     
class manyattention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kqv = trans2attention()
        self.kvq = decoderkvqget()#decoder和encoder交汇
        self.attention1 = attention()
        self.attention2 = attention()
        self.attention3 = attention()
        self.attention4 = attention()
        self.linear = nn.Linear(512,512)


    def forward(self,input,ifdecoder = False,decoder = 0,padding_mask=None):
        if ifdecoder:
            k,v,q = self.kvq(input,decoder)
        else:
            k,v,q = self.kqv(input)
        kbatch = k.shape[0]
        qbatch = q.shape[0]
        vbatch = v.shape[0]
        klen = k.shape[2]
        qlen = q.shape[2]
        vlen = v.shape[2]
        
        k = k.reshape([kbatch,klen,4,-1])
        k = k.transpose(1,2)
        q = q.reshape([qbatch,qlen,4,-1])
        q = q.transpose(1,2)
        v = v.reshape([vbatch,vlen,4,-1])
        v = v.transpose(1,2)
        k1 = k[:,0:1,:,:]
        v1 = v[:,0:1,:,:]
        q1 = q[:,0:1,:,:]
        k2 = k[:,1:2,:,:]
        v2 = v[:,1:2,:,:]
        q2 = q[:,1:2,:,:]
        k3 = k[:,2:3,:,:]
        v3 = v[:,2:3,:,:]
        q3 = q[:,2:3,:,:]
        k4 = k[:,3:4,:,:]
        v4 = v[:,3:4,:,:]
        q4 = q[:,3:4,:,:]
        output1 = self.attention1(k1,v1,q1,padding_mask)
        output2 = self.attention2(k2,v2,q2,padding_mask)
        output3 = self.attention3(k3,v3,q3,padding_mask)
        output4 = self.attention4(k4,v4,q4,padding_mask)
        output = torch.cat([output1,output2,output3,output4],dim = 3)
        output = self.linear(output)
        return output     #多头attention里面有获取和分割KVQ环节

class maskmanyattention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kqv = trans2attention()
        self.maskattention1 = maskattention()
        self.maskattention2 = maskattention()
        self.maskattention3 = maskattention()
        self.maskattention4 = maskattention()
        self.linear = nn.Linear(512,512)

    def forward(self,input,padding_mask=None):
        k,v,q = self.kqv(input)
        kbatch = k.shape[0]
        qbatch = q.shape[0]
        vbatch = v.shape[0]
        klen = k.shape[2]
        qlen = q.shape[2]
        vlen = v.shape[2]
        k = k.reshape([kbatch,klen,4,-1])
        k = k.transpose(1,2)
        q = q.reshape([qbatch,qlen,4,-1])
        q = q.transpose(1,2)
        v = v.reshape([vbatch,vlen,4,-1])
        v = v.transpose(1,2)
        k1 = k[:,0:1,:,:]
        v1 = v[:,0:1,:,:]
        q1 = q[:,0:1,:,:]
        k2 = k[:,1:2,:,:]
        v2 = v[:,1:2,:,:]
        q2 = q[:,1:2,:,:]
        k3 = k[:,2:3,:,:]
        v3 = v[:,2:3,:,:]
        q3 = q[:,2:3,:,:]
        k4 = k[:,3:4,:,:]
        v4 = v[:,3:4,:,:]
        q4 = q[:,3:4,:,:]
        output1 = self.maskattention1(k1,v1,q1,padding_mask)
        output2 = self.maskattention2(k2,v2,q2,padding_mask)
        output3 = self.maskattention3(k3,v3,q3,padding_mask)
        output4 = self.maskattention4(k4,v4,q4,padding_mask)
        output = torch.cat([output1,output2,output3,output4],dim = 3)
        output = self.linear(output)
        return output     #隐藏多头attention里面有获取和分割KVQ环节


class ffn(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(512,1024)
        self.linear2=nn.Linear(1024,512)
        self.rule=nn.ReLU() 
         
    def forward(self,attention):
        x = self.linear1(attention)
        x = self.rule(x)
        x = self.linear2(x)
        return x

class manyffn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512,1024)
        self.linear2 = nn.Linear(512,1024)
        self.linear3 = nn.Linear(512,1024)
        self.linear4 = nn.Linear(512,1024)
        self.linearal = nn.Linear(4096,512)
        self.rule = nn.ReLU()
        
    def forward(self,attention):
        l1 = self.linear1(attention)
        l1 = self.rule(l1)
        l2 = self.linear2(attention)
        l2 = self.rule(l2)
        l3 = self.linear3(attention)
        l3 = self.rule(l3)
        l4 = self.linear4(attention)
        l4 = self.rule(l4)
        lal =torch.cat([l1,l2,l3,l4],dim = 3)
        output = self.linearal(lal)
        return output



class resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.normal1=nn.LayerNorm(512)

    def forward(self,oldinput,attention):
        output = self.normal1(oldinput+attention)
        return output


class embedding(nn.Module):
    def __init__(self,lenid2token):
        super().__init__()
        self.embedding=nn.Embedding(lenid2token,512,padding_idx=0,)

    def forward(self,input):
        output = self.embedding(input)
        return output

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.manyattentionr=manyattention()
        self.resnetr=resnet()
        self.manyffnr=manyffn()
        self.resnetr2=resnet()

    def forward(self,input,encoder_padding_mask):
        manyattention1 = self.manyattentionr(input,padding_mask=encoder_padding_mask)
        resnet1 = self.resnetr(input,manyattention1)
        nextstep = self.manyffnr(resnet1)
        resnet2 = self.resnetr2(resnet1,nextstep)
        return resnet2

class maskattentioninfr(nn.Module):
    def __init__(self):
        super().__init__()
        self.maskmanyattentionr = maskmanyattention()
        self.resnetr = resnet()

    def forward(self,input,decoder_padding_mask):
        maskmanyattention1 = self.maskmanyattentionr(input,decoder_padding_mask)
        resnetr1 = self.resnetr(input,maskmanyattention1)
        output = resnetr1
        return output

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.maskattentioninfrr = maskattentioninfr()
        self.manyattentionr = manyattention()
        self.resnetr = resnet()
        self.manyffnr = manyffn()
        self.resnetr2 = resnet()

    def forward(self,input,encoder,encoder_padding_mask,decoder_padding_mask):
        maskattentioninfr1 = self.maskattentioninfrr(input,decoder_padding_mask)
        manyattention1 = self.manyattentionr(encoder,ifdecoder = True,decoder = maskattentioninfr1,padding_mask=encoder_padding_mask)
        resnetr1 = self.resnetr(maskattentioninfr1,manyattention1)
        manyffnr1 = self.manyffnr(resnetr1)
        output = self.resnetr2(resnetr1,manyffnr1)
        return output     

class fc(nn.Module):
    def __init__(self,lenid2token):
        super().__init__()
        self.linearr = nn.Linear(512,lenid2token)
    def forward(self,input):
        output = self.linearr(input)
        return output

class transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = encoder()
        self.encoder2 = encoder()
        self.encoder3 = encoder()
        self.encoder4 = encoder()
        self.encoder5 = encoder()
        self.decoder1 = decoder()
        self.decoder2 = decoder()
        self.decoder3 = decoder()
        self.decoder4 = decoder()
        self.decoder5 = decoder()

    def forward(self,encdoerinput,decoderinput,encoder_padding_mask, decoder_padding_mask):
        output = self.encoder1(encdoerinput,encoder_padding_mask)
        output = self.encoder2(output,encoder_padding_mask)
        output = self.encoder3(output,encoder_padding_mask)
        output = self.encoder4(output,encoder_padding_mask)
        output = self.encoder5(output,encoder_padding_mask)
        encoderout = output
        output = self.decoder1(decoderinput,encoderout,encoder_padding_mask, decoder_padding_mask)
        output = self.decoder2(output,encoderout,encoder_padding_mask, decoder_padding_mask)
        output = self.decoder3(output,encoderout,encoder_padding_mask, decoder_padding_mask)
        output = self.decoder4(output,encoderout,encoder_padding_mask, decoder_padding_mask)
        output = self.decoder5(output,encoderout,encoder_padding_mask, decoder_padding_mask)
        return output


class reallytrainmodel(nn.Module):
    def __init__(self,lenid2tokenc,lenid2tokene):
        super().__init__()
        self.transformerr = transformer()
        self.embeddingrc = embedding(lenid2tokenc)
        self.embeddingre = embedding(lenid2tokene)
        self.fcr = fc(lenid2tokene)

    def forward(self,encodertoken,nowdecodertoken):
        # [B,1,L] → [B,1,1,L]
        encoder_padding_mask = (encodertoken == 0).unsqueeze(2)
        decoder_padding_mask = (nowdecodertoken == 0).unsqueeze(2)
        embeddingencoder = getencoderemb(self.embeddingrc,encodertoken)
        embeddingdecoder = self.embeddingre(nowdecodertoken)
        embeddingdecoder = embaddposition(embeddingdecoder)
        transformer1 = self.transformerr(embeddingencoder,embeddingdecoder,encoder_padding_mask,decoder_padding_mask)
        ans = self.fcr(transformer1)
        return ans


def position(third,i):
    if i%2==1:
        output = cos(third/(pow(10000,(i-1)/512)))
    else:
        output = sin(third/(pow(10000,i/512)))
    return output

def embaddposition(emb):
    length = emb.shape[2]
    widden = emb.shape[3]
    positionmar = torch.empty((length,widden),dtype = emb.dtype,device = emb.device) 
    for i in  range(emb.shape[2]):
        for j in range(emb.shape[3]):
            positionmar[i][j] = position(i,j)
    positionmar = positionmar.unsqueeze(0)
    positionmar = positionmar.unsqueeze(0)
    return positionmar+emb


def cout(tokenid,id2tokene):
   print(id2tokene[tokenid],end = "",flush = True)

def bulievocab(textc,texte,textcpath,textepath):
    id2tokenc,dictc = xray(textc,textcpath)
    id2tokene,dicte = xray(texte,textepath)
    return dictc,dicte,id2tokenc,id2tokene

def myinput4train(textc,texte,dictc,dicte):

    tokenencoder ,tokendecoder,tokenanswer= trans2token(textc,texte,dictc,dicte)
    return tokenencoder,tokendecoder,tokenanswer


def myinput4inference(textc,dictc,dicte):
    textc = [textc]
    texte = [""]
    tokenencoder,tokendecoder,tokenanswer = trans2token(textc,texte,dictc,dicte)
    return tokenencoder,tokendecoder

def getencoderemb(encdoeremb,encodertoken):
    embeddingencoder = encdoeremb(encodertoken)
    embeddingencoder = embaddposition(embeddingencoder)
    return embeddingencoder


def inference(maxcoutlen,model,encodertoken,nowdecodertoken,id2token):
    with torch.no_grad():
         for s in range(maxcoutlen):
             ans = model(encodertoken,nowdecodertoken)
             tokenid = torch.argmax(ans[0][0][-1]).item()
             if tokenid == 2:
                 break
             cout(tokenid,id2token)
             temptensor = nowdecodertoken.new_full((nowdecodertoken.shape[0],1,1),tokenid)
             nowdecodertoken = torch.cat([nowdecodertoken,temptensor],dim = 2)
    return 0