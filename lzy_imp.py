import numpy as np 
import cv2
import glob 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os 
from torch.autograd import Variable
import time
import matplotlib
#HYPER PARAM for data_loader
AGE_CATEGORIES = 10
#HYPER PARAM for network
IMG_SIZE = 128
AGE_CATEGORY = 10
Z_DIM = 50
#HYPER PARAM for training
MAXITER = 100000

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
### Start of the data loading area

class data_reader():
    def __init__(self):
        self.load_data()
        random.shuffle(self.data)
        self.pos = 0

    def process_age(self, age):
        if 0 <= age <= 5:
            age = 0
        elif 6 <= age <= 10:
            age = 1
        elif 11 <= age <= 15:
            age = 2
        elif 16 <= age <= 20:
            age = 3
        elif 21 <= age <= 30:
            age = 4
        elif 31 <= age <= 40:
            age = 5
        elif 41 <= age <= 50:
            age = 6
        elif 51 <= age <= 60:
            age = 7
        elif 61 <= age <= 70:
            age = 8
        else:
            age = 9
        return age

    def load_data(self):
        print('Loading data...')
        data = []
        for i in glob.glob('./data/UTKFace/*.jpg'):    #通过globe生成文件列表
            img = cv2.imread(i)    #采用cv2读取图像
            img = cv2.resize(img, (128, 128))    #将图像裁剪为 128*128
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            i = i.replace('\\','/').split('/')[-1]    #将目的路径的所有双转意符号转化为‘/’，通过列表分割，选取最后一位（文件名）
            i = i.split('_')    #将文件名用'_'分割为三段，i[0],i[1]分别是年龄和性别
            age = int(i[0])
            gender = int(i[1])
            age = np.eye(AGE_CATEGORIES)[self.process_age(age)]    #选取对角矩阵的第i行
            gender = np.eye(2)[gender]
            data.append([img, age, gender])    #添加在数据的最后面
        self.data = data    #全部读取完成后将对应的数据保存在类的data中
        print('Load finished.')

    def process_image(self,img):
        img = np.float32(img) / 127.5 - 1.
        img = torch.from_numpy(np.transpose(img, (0,3,1,2)))
        return img 

    def get_next_batch(self, bsize):
        if self.pos + bsize > len(self.data):
            random.shuffle(self.data)
            self.pos = 0

        batch = self.data[self.pos: self.pos+bsize]
        self.pos += bsize

        img_batch, age_batch, gender_batch = list(zip(*batch))

        img_batch = self.process_image(img_batch)
        age_batch = torch.from_numpy(np.float32(age_batch))
        gender_batch = torch.from_numpy(np.float32(gender_batch))
        
        return img_batch, age_batch, gender_batch

### Finish of the data loading area


### Start of the network area

#FINISH ENCODERNET
class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet,self).__init__()
        #原文说这里should modify the network structure for better training
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2)
        self.fc = nn.Linear(in_features=1024, out_features=50)  #Z_DIM
        self.activation = nn.ReLU()

    def forward(self,x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = x.view(-1,self.num_flat_features(x))
        x = self.fc(x)
        return torch.tanh(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
#FINISH ENCODERNET

#FINISH DECODERNET
class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet,self).__init__()
        #原文说这里should modify the network structure for better training
        self.fc = nn.Linear(in_features=200, out_features=4 * 4 * 1024)  # Z_DIM
        self.conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=5, stride=1, padding=2)
        # attention
        self.a1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.a2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.a3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.activation = nn.ReLU()
        self.activation_att = nn.Sigmoid()

    def forward(self, x, age, gender, img):
        # gender = gender.unsqueeze(1).repeat(1,1,25).view(100,-1)
        # age = age.unsqueeze(1).repeat(1,1,10).view(100,-1)
        age = self.tile(age,1,10)
        gender = self.tile(gender,1,25)
        x = torch.cat([x,age,gender], 1)
        x = self.activation(self.fc(x))
        x = x.view(-1, 1024, 4, 4)
        #        x = tf.reshape(x,[-1,4,4,1024])
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        att = x
        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = self.activation(self.conv7(x))
        x = torch.tanh(x)
        att = self.activation(self.a1(att))
        att = self.activation(self.a2(att))
        att = self.activation_att(self.a3(att))
        return x, img * att + x * (1.-att), att

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def tile(self,a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index).to(device)
#FINISH DECODERNET

class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ,self).__init__()
        self.fc1 = nn.Linear(in_features=50, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class DiscriminatorPatch(nn.Module):
        #     """
        # (100, 128, 128, 3)
        # (100, 64, 64, 32)
        # (100, 32, 32, 64)
        # (100, 16, 16, 128)
        # (100, 8, 8, 256)
        # (100, 4, 4, 512)
        # (100, 8192)
        # (100, 512)
        # (100, 10)
        #     """
    def __init__(self):
        super(DiscriminatorPatch,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2,padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2,padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2,padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2,padding=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2,padding=2)
        self.fc = nn.Linear(in_features=8192, out_features=512)
        self.fc_age = nn.Linear(in_features=512, out_features=10)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        discrminate = self.conv5(x)
        x = self.activation(self.conv6(x))
        x = x.view(-1,self.num_flat_features(x))
        x = self.activation(self.fc(x))
        age = self.fc_age(x)
        return discrminate, age
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AgeClassifier(nn.Module):
    # (100, 50)
    # (100, 128)
    # (100, 64)
    # (100, 10)
    def __init__(self):
        super(AgeClassifier,self).__init__()
        self.fc1 = nn.Linear(in_features=50, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.activation = nn.ReLU()

    def forward(self, x, reverse_grad):
        if reverse_grad:
            x = -x
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x 


def disLoss(d_real, d_fake):
    # use Mean Square Gan loss
    #d_loss
    d_loss_real = torch.mean(torch.pow((d_real - torch.ones_like(d_real)),2))
    d_loss_fake = torch.mean(torch.pow((d_fake - torch.zeros_like(d_fake)),2))
    d_loss = (d_loss_real + d_loss_fake) * 0.5
    #g_loss
    g_loss = torch.mean(torch.pow((d_fake - torch.ones_like(d_fake)),2))
    return d_loss, g_loss

def ageLoss(pred, label):
    pred_softmax = F.softmax(pred)
    label_cross = label * torch.log(pred_softmax)
    loss = - torch.sum(label_cross,1)
    loss = torch.mean(loss)
    return loss

def tvLoss(x,TVLoss_weight = 1):
    def _tensor_size(t):
        return list(x.size())[1]*list(x.size())[2]*list(x.size())[3]
    batch_size = x.size()[0]
    h_x = list(x.size())[2]
    w_x = list(x.size())[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return _tensor_size(x)*list(x.size())[0]*TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def mseLoss(pred, label):
    loss = torch.mean(torch.abs(pred - label))
    return loss 

### Finish of the network area

def printLosses(losses, i, eta):
    loss_img, loss_dis_z_d, loss_dis_z_g, loss_dis_img_d, loss_dis_img_g, loss_c, loss_c_rev, loss_ae_d, loss_ae_g = losses

    print('ITER:%d\tIMG:%.4f\tDZ:%.4f\tGZ:%.4f\tDIMG:%.4f\tGIMG:%.4f\tC1:%.4f\tC2:%.4f\tAE:%.4f\tAE2:%.4f\tETA:%s'%\
        (i, loss_img, loss_dis_z_d, loss_dis_z_g, loss_dis_img_d, loss_dis_img_g, loss_c, loss_c_rev, loss_ae_d, loss_ae_g, eta.get_ETA(i)))

class ETA():
    def __init__(self,max_value):
        self.start_time = time.time()
        self.max_value = max_value
        self.current = 0

    def start(self):
        self.start_time = time.time()
        self.current = 0

    def sec2hms(self,sec):
        hm = sec//60
        s = sec%60
        h = hm//60
        m = hm%60
        return h,m,s

    def get_ETA(self,current,is_string=True):
        self.current = current
        time_div = time.time() - self.start_time
        time_remain = time_div * float(self.max_value - self.current) / float(self.current + 1)
        h,m,s = self.sec2hms(int(time_remain))
        if is_string:
            return '%d:%d:%d'%(h,m,s)
        else:
            return h,m,s

class AIM(nn.Module):
    def __init__(self):
        super(AIM,self).__init__()
        self.encoder = EncoderNet().to(device)
        self.decoder = DecoderNet().to(device)
        self.dis_z = DiscriminatorZ().to(device)
        self.age_classifier = AgeClassifier().to(device) 
        self.dis_img = DiscriminatorPatch().to(device)
        
    def generate(self,x,age_batch,gender_batch,img):
        res, res_att, _ = self.decoder(self.encoder(x), age_batch, gender_batch, img)
        # can choose either res or res_att as output
        res_att_cpu = torch.Tensor.cpu(res_att)
        res = (res_att_cpu.detach().numpy() + 1) * 127.5
        res = np.uint8(res)
        return res 

class All_Loss(nn.Module):
    def __init__(self):
        super(All_loss,self).__init__()
        
    def forward(self, losses, loss_grad, tape,AIM_model):
        z_enc = AIM_model.encoder(img_batch)
        img_fake, img_fake_att, att = AIM_model.decoder(z_enc, age_batch, gender_batch, img_batch)

        # feature discriminator
        dis_z_fake = AIM_model.dis_z(z_enc)
        dis_z_real = AIM_model.dis_z(2*(torch.rand(z_enc.shape)-0.5*torch.ones(z_enc.shape)))

        # age classifier
        age_pred = AIM_model.age_classifier(z_enc, reverse_grad=False)
        age_pred_r = AIM_model.age_classifier(z_enc, reverse_grad=True)

        # image discriminator
        dis_img_fake, age_fake = AIM_model.dis_img(img_fake)
        dis_img_fake_att, age_fake_att = AIM_model.dis_img(img_fake_att)
        dis_img_real, age_real = AIM_model.dis_img(img_batch)

        # build losses 
        # reconstruction loss
        loss_img = mseLoss(img_batch, img_fake) + 0.001*tvLoss(img_fake) + 0.001*tvLoss(att) + 0.01*torch.mean(torch.pow(att,2))
        loss_dis_z_d, loss_dis_z_g = disLoss(dis_z_real, dis_z_fake)
        loss_dis_img_d, loss_dis_img_g1 = disLoss(dis_img_real, dis_img_fake)
        loss_dis_img_d, loss_dis_img_g2 = disLoss(dis_img_real, dis_img_fake_att)
        loss_dis_img_g = loss_dis_img_g1 + loss_dis_img_g2

        # c loss
        loss_c = ageLoss(age_pred, age_batch)
        loss_c_rev = ageLoss(age_pred_r, torch.ones(age_batch)/AGE_CATEGORY)

        # ae loss
        loss_ae_d = ageLoss(age_real, age_batch)
        loss_ae_g = ageLoss(age_fake_att, age_batch) + ageLoss(age_fake, age_batch)

        losses = [loss_img, loss_dis_z_d, loss_dis_z_g, loss_dis_img_d, loss_dis_img_g, loss_c, loss_c_rev, loss_ae_d, loss_ae_g]
        weights = [1., 0.001, 0.001, 0.1, 0.1, 0.01, 0.01, 0.1, 0.1]
        loss_grad = [w*l for w,l in zip(weights, losses)]
        return losses, loss_grad, tape

def all_Loss(img_batch, age_batch, gender_batch,AIM_model):
        z_enc = AIM_model.encoder(img_batch)
        img_fake, img_fake_att, att = AIM_model.decoder(z_enc, age_batch, gender_batch, img_batch)

        # feature discriminator
        dis_z_fake = AIM_model.dis_z(z_enc)
        dis_z_real = AIM_model.dis_z((2*(torch.rand(z_enc.shape)-0.5*torch.ones(z_enc.shape))).to(device))

        # age classifier
        age_pred = AIM_model.age_classifier(z_enc,reverse_grad=False)
        age_pred_r = AIM_model.age_classifier(z_enc, reverse_grad=True)

        # image discriminator
        dis_img_fake, age_fake = AIM_model.dis_img(img_fake)
        dis_img_fake_att, age_fake_att = AIM_model.dis_img(img_fake_att)
        dis_img_real, age_real = AIM_model.dis_img(img_batch)

        # build losses 
        # reconstruction loss
        loss_img = mseLoss(img_batch, img_fake) + 0.001*tvLoss(img_fake) + 0.001*tvLoss(att) + 0.01*torch.mean(torch.pow(att,2))
        loss_dis_z_d, loss_dis_z_g = disLoss(dis_z_real, dis_z_fake)
        loss_dis_img_d, loss_dis_img_g1 = disLoss(dis_img_real, dis_img_fake)
        loss_dis_img_d, loss_dis_img_g2 = disLoss(dis_img_real, dis_img_fake_att)
        loss_dis_img_g = loss_dis_img_g1 + loss_dis_img_g2

        # c loss
        loss_c = ageLoss(age_pred, age_batch)
        loss_c_rev = ageLoss(age_pred_r, torch.ones_like(age_batch)/AGE_CATEGORY)

        # ae loss
        loss_ae_d = ageLoss(age_real, age_batch)
        loss_ae_g = ageLoss(age_fake_att, age_batch) + ageLoss(age_fake, age_batch)

        losses = [loss_img, loss_dis_z_d, loss_dis_z_g, loss_dis_img_d, loss_dis_img_g, loss_c, loss_c_rev, loss_ae_d, loss_ae_g]
        weights = [1., 0.001, 0.001, 0.1, 0.1, 0.01, 0.01, 0.1, 0.1]
        loss_grad = [w*l for w,l in zip(weights, losses)]
        return losses, loss_grad

if __name__ == "__main__":
    if not os.path.exists('./model/'):
        os.mkdir('./model/')
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    AIM_model = AIM()
    AIM_model = AIM_model.to(device)
    optimi = optim.Adam(AIM_model.parameters(),0.0001)
    torch.save(AIM_model.state_dict(),"./model/aimmodel.pth")

    reader = data_reader()  #新建一个data_reader类

    eta = ETA(MAXITER+1)
    for i in range(MAXITER+1):
        optimi.zero_grad()
        img_batch, age_batch, gender_batch = reader.get_next_batch(100)
        img_batch = img_batch.to(device)
        age_batch = age_batch.to(device)
        gender_batch = gender_batch.to(device)

        losses, loss_grad = all_Loss(img_batch, age_batch, gender_batch, AIM_model)
        print(losses,loss_grad)
        

        # if i%10==0:
        #   network.printLosses(losses, i, eta)
        if i%10==0:
            printLosses(losses,i,eta)

        if i%1000==0:
            # visualize every 1000 iters
            for k in range(10):

                age_batch = np.zeros([age_batch.shape[0], 10],np.float32,)
                age_batch[:,k] = 1
                age_batch = torch.from_numpy(age_batch).to(device)
                res = AIM_model.generate(img_batch, age_batch, gender_batch, img_batch)
                print(res.max())
                print(res.min())
                img_batch_cpu = torch.Tensor.cpu(img_batch)
                img_r = np.uint8((img_batch_cpu+1.)*127.5)
                img_r_pic = np.transpose(img_r, (0,2,3,1))
                res_pic = np.transpose(res,(0,2,3,1))
                for j in range(len(res)):
                    img_r_pic[j] = cv2.cvtColor(img_r_pic[j], cv2.COLOR_RGB2BGR)
                    cv2.imwrite('./results/%d_%d_r.jpg'%(i,j), img_r_pic[j])
                for j in range(len(res)):
                    res_pic[j] = cv2.cvtColor(res_pic[j], cv2.COLOR_RGB2BGR)
                    cv2.imwrite('./results/%d_%d_%d.jpg'%(i,j,k), res_pic[j])

        if i%2000==0 and i>0:
            print("current i: ",i)
            torch.save(AIM_model.state_dict(),"./model/aimmodel.pth")
    # print(reader)
    # print("img_batch size:",img_batch.size," shape:", img_batch.shape)
    # print("age_batch size:",age_batch.size," age_batch shape: ",age_batch.shape)
    # print("gender_batch size:",gender_batch.size," gender_batch shape: ",gender_batch.shape)

    # encoder = EncoderNet()
    # decoder = DecoderNet()
    # out = encoder(img_batch)
    # print("encoder(img_batch) shape: ",out.shape)

    # res, res_att, _ = decoder(out, age_batch, gender_batch, img_batch)
    # print("res shape: ",res.shape)
    # print("res_att  shape: ",res_att.shape)
    # print("_  shape: ",_)


    # stat_to_restore = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    # model_path = "./saver/modelepoch"+epoch+".pth"
    # torch.save(stat_to_restore, model_path)