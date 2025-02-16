'''
RL Agent to change image brightness according
to image(state) to get maximum performance from
a pre-trained network, in this case YOLO
Agent network is ResNet18 with REINFORCE Algorithm
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#from Policy2 import *   ##### modified Policy2.py to Policy3.py
# from Policy2 import *
from utils import *
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
from resnet_policy import resnet18
from Yolo_detect import Detector
import pandas as pd

'''
The argparse module's support for command-line interface is built around an instance of argparse.ArgumentParser. It is a container for argument specifications.
The .add_argument() method attaches individual argument specifications to the parser.
The .parse_agrs() method runs the parser and places the extracted data in a argparse.Namespace object.
'''
parser = argparse.ArgumentParser(description='PyTorch REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--load_model',type=int,default=0,metavar='N',
                    help='whether to use the saved model or not, to resume training')
parser.add_argument('--weights',default='Weights_det.pt',
                    help='Path to the weights.')
parser.add_argument('--optimizer',default='Optimizer_det.pt',
                    help='Path to the Optimizer.')
parser.add_argument('--threshold',default=0.85,type=float,
                    help='Threshold for the Similarity Index')
parser.add_argument('--show',default=0,
                    help='To show the images before and after transformation')
parser.add_argument('--episodes',type=int,default=10,
                    help='Number of episodes')
parser.add_argument('--lr',type=float,default=1e-6,
                    help='Learning rate')
parser.add_argument('--std',default=0,
                    help='To print standard deviation of the probabilities')
parser.add_argument('--confidence',type=float,default=0.5,
                    help='Confidence Threshold for Object Detection(YOLO)')
parser.add_argument('--nms_thresh',type=float,default=0.3,
                    help='Non Maximal Suppression Threshold for YOLO')
parser.add_argument('--reso',type=int,default=416,
                    help='Resolution of Image to be fed into YOLO for detection keeping the Aspect Ration constant')
parser.add_argument('--image_filepath',type=str,default='VOCdevkit/VOC2007/',
                    help='VOC Dataset')
parser.add_argument('--iou_threshold',type=float,default=0.5,
                    help='Threshold for IOU to determine if object is detected or not')
parser.add_argument('--alpha',type=float,default=0.5,
                    help='IOU Weight for reward --> r=alpha*(iou)+(1-alpha)*F1')
parser.add_argument('--epoch',type=int,default=1,
                    help='Number of epochs')


args = parser.parse_args()

df=pd.read_csv('labels.csv')
image_filepath=args.image_filepath+str('JPEGImages/') #train images are here
annotations_filepath=args.image_filepath+str('Annotations/') # test images are here
num_images=len(os.listdir(image_filepath))  # total number of images in the dataset
action_table_synth=np.linspace(0.05,2,40) # to synthetically change the images
action_table=1/action_table_synth # the optimal actions are reciprocal of the factor of the synthesized image
# 0 means complete dark,2 means complete bright, 1 means the original image

###########################################################################

policy = resnet18()  # can also be resnet34, resnet50, resnet101, resnet152
# Constructs and returns a ResNet-18 model
if args.load_model==0:
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()
classes = load_classes('data/coco.names')

CUDA = torch.cuda.is_available()
''' CUDA stands for Compute Unified Device Architecture. CUDA is a parallel computing platform and application programming interface that allows software to use certain types of graphics processing units for general purpose processing, an approach called general-purpose computing on GPUs. '''
if CUDA:
   print('CUDA available, setting GPU mode')
   print('GPU Name:',torch.cuda.get_device_name(0))
   print('Memory Usage:')
   # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
   print('Allocated:', torch.cuda.memory_allocated(0)/1024**3, 'GB')
   print('Cached:   ', torch.cuda.memory_cached(0)/1024**3, 'GB')
   # print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
   policy.cuda()
   # DOUBT : What does the line "policy.cuda" mean? Does it implement the policy on the GPU and that's it?


print('Loading the model if any')
print(args.load_model)
if args.load_model==1:
    policy.load_state_dict(torch.load(args.weights))
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    # optim is a package implementing various optimization algorithms.
    '''Adam algorithm is a replacement optimization algorithm for stochastic gradient descent for training deep learning models.'''
    optimizer.load_state_dict(torch.load(args.optimizer))
    print(args.weights)
    print(args.optimizer)

def save_model():
    '''
    This function saves the weights used in the policy and the optimizer. The torch.save method saves an object to a disk file.
    A state_dict is an integral entity if you are interested in saving or loading models fromm PyTorch.
    '''
    print('Saving the weights')
    torch.save(policy.state_dict(),args.weights)
    torch.save(optimizer.state_dict(),args.optimizer)

def select_action(state):
    '''
    This function selects an action given a state.
    Unsqueeze is a method to change the tensor dimensions, such that operations such as tensor multiplication can be possible.
    '''
    state = torch.from_numpy(state).float().unsqueeze(0)
    if CUDA:
        state = state.cuda()
    probs1 = policy(state)
    #print(probs1)
    m1 = Categorical(probs1)
    if args.std:
        print(probs1.std())
    action1 = m1.sample()
    policy.saved_log_probs1.append(m1.log_prob(action1))
    #print(m1.log_prob(action1))
    return action1.item()

def finish_episode():
    # print('Finishing Episode')
    R = 0
    policy_loss1 = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    if CUDA:
        rewards = rewards.cuda()
    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs1, rewards):
        policy_loss1.append(-log_prob * reward)
        #print(policy_loss1)

    optimizer.zero_grad()
    policy_loss1 = torch.cat(policy_loss1).sum()
    #print(policy_loss1)
    #print(policy_loss2)
    policy_loss = policy_loss1
    #print(policy_loss)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs1[:]



def main():
    image_list = os.listdir(image_filepath)
    #num_images=10
    reward_epoch=[]
    for epoch in range(args.epoch):
        image_list = shuffle_arr(image_list) #shuffle_arr from utils.py shuffles the array randomly
        reward_arr=[]
        # for episodes in range(args.episodes):
        for episodes in tqdm(range(num_images)):
        # for episodes in tqdm(range(3)):
            img_name = image_list[episodes]
            # index_img = np.random.uniform(low = 0, high = num_images, size = (1,)).astype(int) # random number between 0 and num_images eg: 435
            # img_name = os.listdir(image_filepath)[index_img[0]] # eg: 000005.jpg
            img = Image.open(image_filepath+img_name)
            orig_img_arr = np.array(img)
            orig_img_arr_lr, w, h = letterbox_image(orig_img_arr,(args.reso,args.reso)) #resized image by keeping aspect ratio same
            x = Image.fromarray(orig_img_arr_lr.astype('uint8'), 'RGB')
            orig_img_arr=np.array(x)

            ## mean size of image is (384,472,3)
            # img = read(folder+str(index_img[0])+'.png',64)
            #synthesize the image
            change_img , act = synthetic_change(img,action_table_synth,1)
            #convert to array
            change_img_arr = np.array(change_img)
            change_img_arr, w, h= letterbox_image(change_img_arr,(args.reso,args.reso))
            x = Image.fromarray(change_img_arr.astype('uint8'), 'RGB')
            change_img_arr=np.array(x)
            #img_arr = np.array(img)
            state=change_img_arr/255
            state = (np.reshape(state,(3,state.shape[0],state.shape[1]))) # to get into pytorch mode
            #take action acording to the state
            bright_action=select_action(state)
            agent_act=action_table[bright_action]
            #synthetically changed image is rectified by the agent
            new_image = change_brightness(change_img,action_table[bright_action])
            # new_image = change_color(new_image,action_table[color_action])

            # feed the changed image to detector
            new_image_arr=np.array(new_image)
            new_image_arr, w, h = letterbox_image(new_image_arr,(args.reso,args.reso))
            x = Image.fromarray(new_image_arr.astype('uint8'), 'RGB')
            new_image_arr=np.array(x)
            if args.show:
                plt.imshow(orig_img_arr)
                plt.title('Original')
                plt.show()
                plt.imshow(change_img_arr)
                plt.title('Modified image')
                plt.show()
                plt.imshow(new_image_arr)
                plt.title('Agent modified image')
                plt.show()
            # print('Getting detections')
            detector = Detector(args.confidence,args.nms_thresh,args.reso,new_image_arr)
            d=detector.detector()

            # get ground truths
            ground_truth_df = df[df['filename'] == img_name]
            ground_truth_arr=[]
            for i in range(len(ground_truth_df)):
                ground_truth_arr.append(ground_truth_df.iloc[i][1:])
            ground_truth_arr=np.array(ground_truth_arr)

            # rearrange gnd truth array to [xmin,ymin,xmax,ymax,width,height,class] and get resized bboxes
            resized_gnd_truth_arr=np.copy(ground_truth_arr)
            for i in range(len(ground_truth_arr)):
                arr=ground_truth_arr[i][:4]
                t=getResizedBB(arr,h,w,orig_img_arr.shape[0],orig_img_arr.shape[1],args.reso)
                resized_gnd_truth_arr[i][:4]=np.array(t)

            # rearrange predicted arrays and get class name from number
            pred = np.array(d)
            pred_arr=[]
            for i in range(len(pred)):
                arr=list(pred[i][1:5])
                arr.append(classes[int(pred[i][-1])])
                arr=np.array(arr)
                pred_arr.append(arr)
            pred_arr=np.array(pred_arr)

            # get F1 Score
            # get IOU average of all detected objects


            TP,FP,FN,iou = get_F1(resized_gnd_truth_arr,pred_arr,args.iou_threshold)
            # reward=np.mean(IOU+F1_score) #to make sure everything is in 0-1 range
            recall = TP/(TP+FN+eps)
            precision = TP/(TP+FP+eps)
            F1 = 2*recall*precision/(precision+recall+eps)
            if len(iou)>0: #### if no detections then iou=[]
                iou_reward = np.mean(iou)
            else:
                iou_reward = 0
            reward = args.alpha*(iou_reward)+(1-args.alpha)*F1
            reward_arr.append(reward)
            # print('Episode:%d \t Reward:%f'%(episodes,reward))
            policy.rewards.append(reward)
            finish_episode()  # does all backprop
            print_arg=False
            if print_arg:
                print('F1:%f'%(F1))
                print('Reward:%f'%(reward))
                print('Action:%f'%(act))
                print('Agent Action:%f'%agent_act)
                print('Ideal action:%f'%(1/act))
                print()

        save_model()
        print('#'*50)
        print()
        print('Epochs:%d'%epoch)
        print('Mean reward:%f'%(np.mean(reward_arr)))
        print()
        print('#'*50)
        reward_epoch.append(np.mean(reward_arr))
    print('Reward array:',reward_epoch)

if __name__ == '__main__':
    main()
