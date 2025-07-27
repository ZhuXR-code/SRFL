import src.inner_product.single_input_fe.elgamal_ip.Dfe
import numpy as np
import unittest
import time

class TestElGamalInnerProduct(unittest.TestCase):

    def test_fin_result(self):

        fe =src.inner_product.single_input_fe.elgamal_ip.Dfe
        pk, sk = fe.KeyGen(1024, 5)
        pk1,sk1=pk[0],sk[0] #server pk,sk
        pk2,sk2= [pk[1],pk[2],pk[3]], [sk[1],sk[2],sk[3]] #clint pk sk
       
        y = [1, 1, 1]#y的个数要跟encrtyptor的个数一致，encryptor个数为3，y的长度也为3
        x = [1, 2, 3]#
        
        ctr=0
        aux='dfe'
        index=1
        n=3
        Y1=fe.KeyDerive(pk1,sk2,pk2,ctr,y,aux,index,n)
        print(Y1[0])
        Skf=fe.Aggkey(Y1[1])
        print('skf是',Skf)
        if Skf== Y1[0]:
            print ('skf添加噪声后相机结果相同')
        else:
            print('fales')
        t1=time.time()
        x1=fe.Encrypt(pk1,sk2,ctr,x,aux,index)
        t2=time.time()
        t3=t2-t1
        print('encrypt time',t3)
        t4=time.time()
        result=fe.Decrypt(sk1,pk1,Skf,x1[0],Y1[2])
        print('添加噪声解密的结果是',result)
         
        if result !=None:
            result2=fe.userdec(result,sk2,ctr,aux,x1[1])
            print('最终结果是',result2)
        else:
            return('None1')    
        t5=time.time()
        t6=t5-t4
        print('decrypt time',t6)
if __name__ == "__main__":
    unittest.main()
