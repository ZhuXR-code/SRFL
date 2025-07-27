from charm.toolbox.integergroup import IntegerGroupQ, integer
from typing import List, Dict, Tuple
from FE_inner_product_master.src.helpers.additive_elgamal import AdditiveElGamal, ElGamalCipher
from FE_inner_product_master.src.helpers.helpers import reduce_vector_mod, get_int
#from src.errors.wrong_vector_for_provided_key import WrongVectorForProvidedKey
import charm
import numpy as np
import hashlib

IntegerGroupElement = charm.core.math.integer.integer
ElGamalKey = Dict[str, IntegerGroupElement]
#生成群
debug = True
p = integer(
    148829018183496626261556856344710600327516732500226144177322012998064772051982752493460332138204351040296264880017943408846937646702376203733370973197019636813306480144595809796154634625021213611577190781215296823124523899584781302512549499802030946698512327294159881907114777803654670044046376468983244647367)
q = integer(
    74414509091748313130778428172355300163758366250113072088661006499032386025991376246730166069102175520148132440008971704423468823351188101866685486598509818406653240072297904898077317312510606805788595390607648411562261949792390651256274749901015473349256163647079940953557388901827335022023188234491622323683)
elgamal_group = IntegerGroupQ()
elgamal = AdditiveElGamal(elgamal_group, p, q)
elgamal_params = {"group": elgamal_group, "p": int(p)}

#创建哈希函数
def H1(data: str, p: integer) -> integer:
    hash_object = hashlib.sha256(data.encode('utf-8'))
    return integer(int(hash_object.hexdigest(), 16) % p)
#输入公共参数，输出公钥密钥
def KeyGen(security_parameter: int, vector_length: int) -> Tuple[List[ElGamalKey], List[ElGamalKey]]:
    master_public_key = [None] * vector_length
    master_secret_key = [None] * vector_length
    for i in range(vector_length):
        (master_public_key[i], master_secret_key[i]) = elgamal.keygen(secparam=security_parameter)
    return master_public_key, master_secret_key
#生成功能密钥
def KeyDerive(master_public_key:List[ElGamalKey],master_secret_key2: List[ElGamalKey], master_public_key2:List[ElGamalKey],  ctr: int, y: List[int], aux: str,index1:int,n:int) -> integer:
    y = reduce_vector_mod(y, elgamal_params['p'])   # y的个数要跟encrtyptor的个数一致，encryptor个数为3，y的长度也为3
    ctr += 1      # 计数器增加1。
    sum1=0
    dk=[]
    dk_noise = []
    if index1 == 1:
        for i in range(len(y)):
            s_i = master_secret_key2[i]
            pk2i = master_public_key2[i]
            
            s=s_i['x']
            key =master_public_key['h'] ** (s_i['x'])
            r=H1(f"{key}{ctr}{aux}", elgamal_params['p'])
            #分散求dk
            dki=r*y[i]
            dki_noise = r#+r+r
            sum1=sum1+dki
            for j in range(n):
                if i<j:
                    pk2j= master_public_key2[j]
                    key=pk2j['h']**s_i['x']
                    sumij= H1(f"{key}", elgamal_params['p'])
                    dki=dki+sumij
                    dki_noise = dki_noise +sumij
                   # print('第',i,'轮，sum',i,j)
            for j in range(n):
                if i> j:
                    pk2j= master_public_key2[j]
                    key=pk2j['h']**s_i['x']
                    sumij= H1(f"{key}", elgamal_params['p'])
                    dki=dki-sumij
                    dki_noise = dki_noise - sumij
                   # print('第',i,'轮，sum',i,j) 
            dk.append(dki)
            dk_noise.append(dki_noise)
        len_y=len(y)#输出y添加噪声后的结果
        #for i in range(len_y):
         #   y.insert(2*i+1,1)
        #for i in range(len_y):
         #   y.insert(3*i+2,1)
        #print('添加噪声后的y是',y)        
    return(sum1,dk,dk_noise,y)                
def Aggkey(dk_i):
    skf = integer(0)# 初始化一个'skf'，初始值为整数0
    for i in range(len(dk_i)):
        skf=skf+dk_i[i]
    return(skf)

def Encrypt(master_public_key: List[ElGamalKey], master_secret_key2: List[ElGamalKey],ctr: int,  X : List[int], aux: str,index1:int)-> integer:
    g=master_public_key['g']
    C = []
    ctr += 1 
    len_x=len (X)
    j=0
    key11=[] 
    #for i in range(len_x):#将数据扩充一位
    s_i = master_secret_key2  
    key= H1(f"{s_i['x']}", elgamal_params['p'])
    key1=int(key)

    key1 = str(key1)
    first2key = key1[:1]
    key1 = int(first2key)
    #while key1 >= 1000:
        #key1 //= 10
            
    key11.append(key1)
    #print('x添加的噪声是',key1)
            
            
            #print(type(key),type(key1))
            #X.insert(2*i+1,key1)

    #X.append(key1)
    #X.append(0)
    X = [x+key1 for x in X ]
    #for i in range(len_x):#将数据扩充2位
    
            
    #        X.insert(3*i+2,0)
    #print(X)
    #print('添加噪声过后的x是',X)
    #print('enddddddd')
    #for i in X:
       
        #group =X[i:i+3]#由于添加了2位噪声，将每3位分为一组
        #print('添加噪声过后的x是',X)
        
    sk=master_secret_key2#从0开始，每三位*对应一个clint的sk
    key =master_public_key['h'] ** (sk['x'])
    r=H1(f"{key}{ctr}{aux}", elgamal_params['p'])
         
    C = [g**r*(master_public_key['h']**x) for x in X]#加密数据 (g^ski)*(pkserve^x)
        #j+=1
    return(C,key11)   

def Decrypt(master_secret_key: List[ElGamalKey],master_public_key: List[ElGamalKey],skf:List[ElGamalKey],C:List[ElGamalKey],y:List[int]) -> integer:
   
    y = reduce_vector_mod(y, elgamal_params['p'])#将y,c的长度固定一样
    #print(C,y)
  
    E = integer(0)


    #print(len(C))
    #print(len(y))
    c2 = np.product([C[i] ** y[i] for i in range(len(C))])
    #print("c2:",c2)
    sum2=1/(master_public_key['g']**skf)
    E = c2 * sum2
    #print(1/master_secret_key3['x'])
    
    #get g^<x,y>
    E = pow(E,1/master_secret_key['x'])#E**(1/master_secret_key3['x'])
    
    #get <x,y>
    result = dummy_discrete_log(master_public_key['g'], E, elgamal_params['p'], 400)
    
    return result  

def userdec(result:int, master_secret_key2:List[ElGamalKey],ctr:int,aux:str,y:List[int]):
    
    keysum=0
#  for i in range(len(key)):
 #        print(key[i])
  #  ctr=ctr+1
   # sum=0
   # for i in range(len(key)):
    #     keysum+=key[i]
   # print(result,keysum)    ''' 
    for i in range(len( master_secret_key2)):
            s_i = master_secret_key2[i] 
            
            key= H1(f"{s_i['x']}", elgamal_params['p'])
            key1=int(key)
            

            key1 = str(key1)
            first2key = key1[:1]
            key1 = int(first2key)

            #while key1 >= 1000:
             #       key1 //= 10
            keysum=keysum+y[i]*key1
    if type(result) == int:
        result1 = result-keysum
    else:
        result1=0
    return(result1)



def dummy_discrete_log(a: int, b: int, mod: int, limit: int) -> int:
    """Calculates discrete log of b in the base of a modulo mod, provided the
    result is smaller than limit. Otherwise, returns None

    Args:
        a (int): base of logarithm
        b (int): number from which the logarithm is calculated
        mod (int): modulus of logarithm 
        limit (int): limit within which the result should lie

    Returns:
        int: result of logarithm or None if the result was not found withn the limit
    """
    for i in range(limit):
        if pow(a, i, mod) == b:
            return i
    return None    
'''key_2=(master_public_key['h'])**sk2i["x"]
        r= H1(f"{key_2}{ctr}{aux}", elgamal_params['p'])
        C_i=((master_public_key['h'])**r)*(master_public_key3['h']**X[i])
        C.append(C_i)
    print(type(C))
        
    return (C)''' 
